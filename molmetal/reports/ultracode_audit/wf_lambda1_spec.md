# WF-Lambda-1 spec: `molmetal/scripts/r4_lambda_only_run.py`

## Spec (1 paragraph)

`r4_lambda_only_run.py` is a λ-only ablation driver for the Molecular Lambda
Calculus proof-search that re-uses the `r4_c_full_sweep.py` orchestration skeleton
(`PocketResult` dataclass, `aggregate` / `write_csv` / `write_json` / `write_markdown`
emitters, manifest validation, subprocess worker wrapper, runtime SHA256 fingerprinting)
but **hard-disables** every external reward / evaluator: no `search_docking_reward`,
no QuickVina / QuickVina-GPU / Vina, no AdmetAI, no PoseBusters (no `PB_WEIGHT` set),
no REINVENT4, no synthesis oracle, no symbolic prior refit. It instantiates
`MCTSProofSearch` directly with the same 5-rule click set (`CuAAC, SPAAC,
ThiolEne, Suzuki, AmideCoupling`, see `rules.py:59-63`) and the same 204-tile
extended library (`FRAGMENT_LIBRARY_200_TILES` + click-handle extension) wired by
`lambda_100pocket_sweep.py:471-504`, with the canonical `LIPINSKI` + `PROTEASE_GENERIC`
target (`binding/__init__.py:33`, `types.predicates.LIPINSKI`). The candidate-emission
point is `MCTSProofSearch.search` at `proof_search.py:2370` (`return [state for _, state in
candidates[: self.top_k]]`) and the per-candidate descriptor enrichment that already
fires inside `_expand` (`proof_search.py:2578-2635`, exposes `rule_name` per child),
`_attach_children` (`proof_search.py:2637-2663`), and `_simulate` (`proof_search.py:2376-2477`,
records `path` length → `rollout_depth_hist`) — so the λ-only harness can attribute
each leaf to **(a)** its emitting click rule (β-reduction), **(b)** the β-reduction
depth (`rollout_depth_hist`), **(c)** typed-variable hits via `_satisfies_predicates`
(`proof_search.py:3120-3139`), and **(d)** α-equivalence via the `_MCTSNode.to_dict`
canon-SMILES dedup key (`proof_search.py:1601-1639`); `MetalGeometryPrior` pass/fail
(`metal_geometry.py:598-824`) is added post-search per-leaf on the synthetic 3-D
embed produced by `MoleculeClosedTerm.from_smiles(..., embed_3d=False)` → re-embed
when a pocket is supplied, otherwise evaluated structurally on the SMILES via
`MetalGeometryPrior.apply_prior` with `metal_idx=None` (geometry inferred from
`DEFAULT_METAL_GEOMETRY`). The harness consumes the **first 10 test pairs**
(`test_000..test_009`, `data/crossdocked100_manifest.csv` line 2-11) and emits the
same triple-format reports as `r4_c_full_sweep.py` (CSV `r4_lambda_only.<N>.csv` from
`r4_c_full_sweep.py:254-264`, JSON `r4_lambda_only.<N>.json` from `r4_c_full_sweep.py:277-283`,
Markdown `r4_lambda_only.<N>.md` from `r4_c_full_sweep.py:286-320`) at
`molmetal/reports/r4_lambda_only/<timestamp>/`. Honest framing is mandatory: every
per-candidate column is labelled **MEASURED** (algorithmic, computed from
`proof_search.py` + RDKit + `MetalGeometryPrior`) versus **NOT-MEASURED** (any
chemistry property that would require QuickVina / AdmetAI / PB / synthesis oracle
is omitted, not projected). The script reuses `lambda_100pocket_sweep.run_one_pocket`
(`lambda_100pocket_sweep.py:289-702`) **only** for the search-internal state
construction (seed SMILES, tile library, MCTS object construction); the candidate
record assembly is replaced with a λ-only adapter that drops
`physical_evaluation_status`, `synthesis`, `retained_after_synthesis`,
`selected_for_evaluation`, `lipinski` Lipinski-descriptor binding via
`scripts/lambda_100pocket_sweep._lipinski_pass`, and instead adds the λ-native
columns `alpha_class_id` (canonical SMILES hash via `proof_search.py:1709`),
`beta_depth` (from `rollout_depth_hist`), `rule_name` (from `_MCTSNode.rule_name`),
`typed_satisfied` (from `_satisfies_predicates`), `binding_satisfied` (from
`_binds_target` with `oracle=None, leaf_oracle_call=False`), `metal_geometry_pass`
(`MetalGeometryPrior.prior_loss(...) == 0`), `tanimoto_to_train` (max Morgan-fp
Tanimoto vs `crossdocked100_manifest.csv` rows 11-100 as the training proxy —
`training_set_novelty.py:180` `nearest_tanimoto`), `smiles_canonical`
(`molmetal_lam.tile_lib.canonical_cache.canonicalize`), and `valid_rdkit` (bool of
`Chem.MolFromSmiles(canonical_smiles) is not None`). The Markdown report prefixes
every table with a `## Honest framing` block that calls out **MEASURED** (canonical
SMILES count, α-class count, typed/binding pass rate, MetalGeometryPrior pass
rate, Tanimoto-to-training novelty, wall-clock per pocket) versus **NOT-MEASURED**
(no Vina / PB / Admet / QuickVina called, no SOTA comparable success rate). The
deterministic guarantee is preserved via `--seed 42` + the same
`runtime_fingerprints()` SHA256 fingerprinting used in `r4_c_full_sweep.py:111-120`.

## file:line list

- `molmetal/scripts/r4_c_full_sweep.py:47-72` `PocketResult` dataclass (template for new dataclass with λ-native fields)
- `molmetal/scripts/r4_c_full_sweep.py:81-104` `load_manifest` (re-use for first-10 pair selection)
- `molmetal/scripts/r4_c_full_sweep.py:107-120` `file_digest` + `runtime_fingerprints` (re-use for SHA256 audit)
- `molmetal/scripts/r4_c_full_sweep.py:128-214` `run_one_pocket` (template; replace inner `r = load_runner().run_one_pocket(...)` with direct `MCTSProofSearch` driver)
- `molmetal/scripts/r4_c_full_sweep.py:220-251` `aggregate` (re-emit with λ-native metrics; add `n_unique_alpha_classes`, `mean_beta_depth`, `tanimoto_to_train_mean`, `metal_geometry_pass_rate`)
- `molmetal/scripts/r4_c_full_sweep.py:254-264` `write_csv` (re-use unchanged)
- `molmetal/scripts/r4_c_full_sweep.py:267-283` `clean_json` + `write_json` (re-use unchanged)
- `molmetal/scripts/r4_c_full_sweep.py:286-320` `write_markdown` (template; add `## Honest framing` block at top + λ-native table)
- `molmetal/scripts/r4_c_full_sweep.py:324-356` `execute_job` (subprocess worker wrapper; re-use unchanged)
- `molmetal/scripts/r4_c_full_sweep.py:359-599` `main` (template; drop `--physical-docking`, `--search-docking-reward`, `--synthesis-oracle`, `--symbolic-prior`; force N=10 from manifest offset 0)
- `molmetal/molmetal_lam/search_alg/proof_search.py:1826-2061` `MCTSProofSearch` dataclass (instantiate directly; no `scorer`, no `reward` other than zero-weight aggregator)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2059-2370` `MCTSProofSearch.search` (**CANDIDATE EMISSION POINT** — line 2370 `return [state for _, state in candidates[: self.top_k]]`)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2376-2477` `_simulate` (records `path` and `rollout_depth_hist` per simulation → per-candidate `beta_depth`)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2479-2576` `_select_child` (PUCT selection — not needed for λ-only harness but referenced for context)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2578-2635` `_expand` (records `rule_name` and `tile` per child → per-candidate `rule_name`)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2637-2663` `_attach_children` (constructs `_MCTSNode`; `rule_name` + `tile` fields populated)
- `molmetal/molmetal_lam/search_alg/proof_search.py:2934-3046` `_rollout` (rollout policy — not relevant to λ-only attribution but is where `_leaf_value_history` accumulates)
- `molmetal/molmetal_lam/search_alg/proof_search.py:3087-3114` `_backprop` (MCTS backprop; updates `(N, W)` per node)
- `molmetal/molmetal_lam/search_alg/proof_search.py:3120-3139` `_satisfies_predicates` (**TYPED-VARIABLE HITS** gate; per-candidate `typed_satisfied` boolean)
- `molmetal/molmetal_lam/search_alg/proof_search.py:3141-3159` `_binds_target` (**BINDING GATE** — fingerprint stub; `oracle=None`, `leaf_oracle_call=False` so no docking cost; per-candidate `binding_satisfied` boolean)
- `molmetal/molmetal_lam/search_alg/proof_search.py:3183-3194` `score_final` (post-search candidate ranking — not used by λ-only harness since ranking is by typed+binding pass + novelty)
- `molmetal/molmetal_lam/search_alg/proof_search.py:1521-1591` `_MCTSNode` dataclass (N/W/P/children/`rule_name`/`tile`/`expansion_complete`)
- `molmetal/molmetal_lam/search_alg/proof_search.py:1601-1639` `_MCTSNode.to_dict` (uses canonical SMILES as α-equivalence dedup key)
- `molmetal/molmetal_lam/search_alg/proof_search.py:1709-1729` `_smi_of` (canonical-SMILES lookup; same key used for `alpha_class_id`)
- `molmetal/molmetal_lam/lam_chem/ast.py:78-114` `LamVar` (variable node; printed as plain name → `typed-variable hits` come from `_satisfies_predicates` counting free-var occurrences per type predicate)
- `molmetal/molmetal_lam/lam_chem/ast.py:120-178` `LamAbs` (binder; `var.name` is the typed variable slot — count occurrences across candidates via `_satisfies_predicates`)
- `molmetal/molmetal_lam/lam_chem/ast.py:184-242` `LamApp` (function application = β-redex; `beta_reduce()` = single β-step = one click reaction)
- `molmetal/molmetal_lam/lam_chem/ast.py:38-63` `_alpha_rename` (α-conversion; drives α-equivalence collapse in `_MCTSNode.to_dict`)
- `molmetal/molmetal_lam/lam_chem/rules.py:59-63` `CuAAC` / `SPAAC` / `ThiolEne` / `Suzuki` / `AmideCoupling` (the 5 click rules whose `.name` populates each candidate's `rule_name`)
- `molmetal/molmetal_lam/lam_chem/rules.py:72-88` `CLICK_REACTIONS` registry (canonical name → `ReactionRule`)
- `molmetal/molmetal_lam/reactions/beta_reductions.py:144-184` `ReactionRule` dataclass (`name` + `pattern_smiles` = click-rule signature in λ-terms)
- `molmetal/molmetal_lam/reactions/beta_reductions.py:232-265` `ReactionRule.reduce` (the bi-molecular β-reduction invoked by `_expand._safe_reduce`)
- `molmetal/molmetal_lam/priors/metal_geometry.py:598-824` `MetalGeometryPrior` (**METAL COORDINATION PRIOR**; per-candidate `metal_geometry_pass` = `apply_prior(...) == 0`)
- `molmetal/molmetal_lam/priors/metal_geometry.py:329-347` `GEOMETRY_COORDINATION_NUMBER` + `GEOMETRY_IDEAL_ANGLES` (target angles per geometry)
- `molmetal/molmetal_lam/priors/metal_geometry.py:364-374` `DEFAULT_METAL_GEOMETRY` (atomic number → canonical geometry mapping)
- `molmetal/molmetal_lam/priors/metal_geometry.py:769-824` `MetalGeometryPrior.prior_loss` (apply + diagnostics; weight=1.0 for λ-only pass/fail)
- `molmetal/scripts/lambda_100pocket_sweep.py:289-504` `run_one_pocket` (template; re-use seed SMILES resolution + tile library construction lines `389-433`; replace `search.search(seed, ...)` result wrapper)
- `molmetal/scripts/lambda_100pocket_sweep.py:564-702` MCTS driver body (template; drop docking reward wiring `446-462`, drop synthesis gate `594-596`, replace `all_cand_records` with λ-native columns)
- `molmetal/scripts/lambda_100pocket_sweep.py:710-734` `aggregate` (template for λ-native aggregate keys)
- `molmetal/scripts/lambda_100pocket_sweep.py:737-819` `write_csv` + `write_json` + `write_markdown` (template emitters — same schema as r4_c_full_sweep)
- `molmetal/data/crossdocked100_manifest.csv:1-11` First 10 test pairs (`test_000..test_009`); full CSV is 101 lines (header + 100 pairs); test split offset 0..9 = the WF-Lambda-1 N=10 input
- `molmetal/scripts/training_set_novelty.py:180` `nearest_tanimoto` (compute `tanimoto_to_train` per candidate against rows 11-100 of the manifest as the training-set proxy)
- `molmetal/molmetal_lam/tile_lib/canonical_cache.canonicalize` (call site for canonical-SMILES round-trip; populates `smiles_canonical` + `alpha_class_id`)

## λ-only signal inventory (what is and is not measured)

| Signal | Source file:line | MEASURED / NOT-MEASURED |
|---|---|---|
| Canonical SMILES count per pocket | `proof_search.py:1709-1729` `_smi_of` | MEASURED |
| α-class count (unique canonical SMILES) | `proof_search.py:1601-1639` `_MCTSNode.to_dict` dedup | MEASURED |
| β-reduction depth per candidate | `proof_search.py:2446` `rollout_depth_hist[len(path) - 1]` | MEASURED |
| Click rule index per candidate | `proof_search.py:2637-2663` `_attach_children` writes `_MCTSNode.rule_name` | MEASURED |
| Typed-variable hits (Lipinski predicate) | `proof_search.py:3120-3139` `_satisfies_predicates` → `bool(LIPINSKI(state))` | MEASURED |
| Binding-site pass (fingerprint stub, no oracle) | `proof_search.py:3141-3159` `_binds_target` with `oracle=None, leaf_oracle_call=False` | MEASURED (fingerprint stub only) |
| MetalGeometryPrior pass | `metal_geometry.py:659-767` `MetalGeometryPrior.apply_prior` returns 0 ⇒ pass | MEASURED (structural-only when 3-D embed unavailable) |
| SMILES validity via RDKit | `Chem.MolFromSmiles(canonical_smiles) is not None` | MEASURED |
| Novelty vs training-set Tanimoto | `training_set_novelty.py:180` `nearest_tanimoto` | MEASURED (vs manifest rows 11-100 as training proxy) |
| QuickVina docking score | not invoked | NOT-MEASURED (script never imports `molmetal_lam.sbdd_env.*`) |
| AdmetAI desirability | not invoked | NOT-MEASURED (script never imports `molmetal.validation.admet_runner`) |
| PoseBusters pass rate | not invoked | NOT-MEASURED (script never imports `molmetal.validation.posebusters_runner`) |
| REINVENT4 multi-property | not invoked | NOT-MEASURED |
| Synthesis oracle | not invoked | NOT-MEASURED |
| Symbolic prior refit | disabled via `prior_refit_every=0` | NOT-MEASURED |

## Command surface

```
uv run --no-sync python -m molmetal.scripts.r4_lambda_only_run \
    --manifest molmetal/data/crossdocked100_manifest.csv \
    --output-prefix molmetal/reports/r4_lambda_only/r4_lambda_only \
    --n-pockets 10 --pocket-offset 0 \
    --n-simulations 200 --max-depth 3 --top-k 20 \
    --seeds 42 --job-timeout 600
```

CPU-only by construction (no `MolFromSmiles 3-D embed`, no docking binary, no
`molmetal.validation.admet_runner`, no PoseBusters import). Wall-clock target
≤ 10 minutes per pocket (MCTS-only) so the full N=10 sweep runs in < 2 hours.
