# Molecular Lambda Calculus — Per-Layer Metric Catalogue

Inventory of the 9-layer Molecular Lambda Calculus (MLC) stack under
`molmetal/molmetal_lam/`. Each layer is surveyed from its existing Python
module; only metric definitions and 1-line instrumentation sketches are
proposed — no new modules, no framework edits.

---

### Layer 1 — Atoms (combinators.py)

Data flow: Receives element/Z queries (smiles parse) and produces
frozen `Atom` records with `arity = valence + lone_pairs` plus
`PRIMITIVE_ATOMS` / `METAL_ATOMS` registries.

Metrics (5):

1. ARITY_HIT_RATE: Fraction of `(symbol, oxidation)` lookups that
   resolve into `PRIMITIVE_ATOMS` or `METAL_ATOMS` (no fallback
   constructor). Healthy: ≥ 0.95 over ChEMBL subsets.
   PSEUDO-CODE: `# log: from_smiles: hit={lookup_in_lib}/{n_heavy}`
2. METAL_GEOMETRY_OK: Fraction of metal atoms whose `arity` matches the
   documented coordination number (Pt_II=4, Ru_II=6, Zn_II=4, Ir_III=6,
   Cu_II=4, Au_III=4). Unitless ratio. Healthy: 1.0.
   PSEUDO-CODE: `# if METAL_ATOMS[m].arity != expected_cn: counter["bad"]+=1`
3. SANITY_PASS_RATE: `sanity_check() -> Dict[str,bool]`; all True.
   Unitless. Healthy: 1.0.
   PSEUDO-CODE: `# assert_well_formed(); record report["all_true"]`
4. FALLBACK_ATOMS_PER_PARSE: Mean count of atoms that hit the generic
   `Atom(symbol, Z, default_valence, 0, "")` fallback per SMILES parse.
   Unit: atoms/parse. Healthy: ≈ 0 on curated tile library.
   PSEUDO-CODE: `# append (len(atoms)-n_lookup_hits) to history`
5. ARITY_DISTRIBUTION: Histogram of `Atom.arity` over `from_smiles`
   outputs (counts of valence-1/2/3/4 + lone_pairs added).
   Unit: counts. Healthy: dominated by arity ∈ {1,2,3,4}.
   PSEUDO-CODE: `# tally[atom.arity] += 1; emit on shutdown`

Currently tracked: N (no `history` attribute; only `sanity_check()`
report dict).

Target values: Pt_II arity = 4 ✓; Ru_II arity = 6 ✓; cisplatin
descriptor arity sum = 4 ✓ (verified by existing tests).

---

### Layer 2 — Bonds (application.py)

Data flow: Receives `(Atom, Atom)` pairs + order + kind; produces
`Bond` objects and a `FreeSiteLedger` mutating free-site counters; emits
molecule-like dict via `assemble()`.

Metrics (6):

1. BOND_KIND_DISTRIBUTION: Counts of bonds built per kind
   (`covalent`/`dative`/`aromatic`/`hydrogen`). Unit: counts.
   Healthy: dative dominates for metal complexes, aromatic=6n for
   n-rings.
   PSEUDO-CODE: `# per Bond factory: counters[b.kind] += 1`
2. DATIVE_FRACTION: `dative / (dative + covalent)` ratio per assemble.
   Unitless. Healthy: 1.0 for pure coordination complexes
   (e.g. cisplatin), 0.0 for hydrocarbons.
   PSEUDO-CODE: `# histogram per assemble() call`
3. FREE_SITES_AFTER_ASSEMBLE: `assemble(...)['open_sites']` and
   `is_closed` boolean. Units: integer + bool. Healthy: open_sites=0,
   is_closed=True for closed molecules.
   PSEUDO-CODE: `# record open_sites per assemble result`
4. BOND_VALIDITY_RATE: `all(is_valid(b) for b in bond_list)` count
   from `assemble`. Unitless. Healthy: 1.0.
   PSEUDO-CODE: `# record assemble['valid'] boolean`
5. BUILDER_EXCEPTION_RATE: Frequency of `BondError` raised by
   `Bond.covalent/dative/aromatic/hydrogen`. Unit: 1/s.
   Healthy: ≈ 0 for chemistry-correct input; small nonzero rate is
   signal of upstream bad typing.
   PSEUDO-CODE: `# wrap each factory with try/except BondError counter`
6. AROMATIC_RING_SIZES: Histogram of `len(ring_atoms)` passed to
   `Bond.aromatic`. Unit: ring-size counts. Healthy: 5,6,7 dominate.
   PSEUDO-CODE: `# counters[ring_size] += 1`

Currently tracked: N (no per-call logging; only `free_*_before`
snapshots stored on each `Bond` dataclass).

Target values: cisplatin → 4 dative, open_sites=0 ✓ (existing test
case).

---

### Layer 3 — Molecules (closed_term.py)

Data flow: Receives SMILES (or RDKit Mol) or assembled `Bond` list;
produces `MoleculeClosedTerm` with `is_closed`, `is_beta_normal_form`,
`free_sites`, and `alpha_equivalent` predicates.

Metrics (6):

1. IS_CLOSED_RATE: Fraction of `MoleculeClosedTerm.from_smiles` outputs
   with `is_closed == True`. Unitless. Healthy: ≈ 1.0 for non-radical
   drug-like inputs.
   PSEUDO-CODE: `# in from_smiles: rec["is_closed"] = term.is_closed`
2. IS_BETA_NORMAL_FORM_RATE: Fraction with
   `is_beta_normal_form == True`. Unitless. Healthy: high; lower
   indicates tautomerisable fragments.
   PSEUDO-CODE: `# rec["beta_nf"] = term.is_beta_normal_form`
3. REDEX_HIT_RATE: `has_redex()` True fraction over generated terms.
   Unitless. Healthy: 0.0 for fully reduced closed terms.
   PSEUDO-CODE: `# if term.has_redex(): counters["open"] += 1`
4. REDUCE_STEPS_TO_NF: Histogram of `reduce_once()` calls required to
   reach β-NF. Unit: steps. Healthy: 0–2 for well-formed inputs.
   PSEUDO-CODE: `# while term.has_redex(): term=term.reduce_once(); n+=1`
5. ALPHA_EQUIV_COLLISIONS: Number of distinct terms sharing the same
   canonical SMILES in a batch. Unit: collisions/batch. Healthy: 0
   (one canonical SMILES per α-class).
   PSEUDO-CODE: `# canonical_smi_to_terms[cs] += 1; len(>1)`
6. ATOM_BOND_RATIO: `n_atoms / n_bonds` per term. Unitless. Healthy:
   1.5–2.0 for drug-like; 1.0 for linear polymers.
   PSEUDO-CODE: `# rec["atom_bond_ratio"] = term.n_atoms / max(1, term.n_bonds)`

Currently tracked: N (no `history` field; state lives on the dataclass
itself).

Target values: cisplatin n_atoms=5, n_bonds=4 ✓; water is_closed=True
via implicit-H valence_used ✓ (per existing tests).

---

### Layer 4 — Reactions (beta_reductions.py)

Data flow: Receives 1–2 `MoleculeClosedTerm` reactants; produces a
list of product closed terms per `ReactionRule.reduce`. Registry:
`REACTION_RULES = {CuAAC, SPAAC, SPC, DielsAlder, ThiolEne}`.

Metrics (7):

1. FIRE_RATE_PER_RULE: Fraction of `reduce((a, b))` calls returning
   non-empty list, broken down per rule. Unitless. Healthy: high
   (≈ 0.8+) when reactants carry the right handles.
   PSEUDO-CODE: `# counters[rule.name]["fire"] += int(bool(products))`
2. PRODUCT_COUNT_PER_REDUCE: Histogram of
   `len(products)` per `reduce()` call. Unit: products/fire.
   Healthy: 1–2; multi-product indicates regioisomers (SPAAC 1,4/1,5).
   PSEUDO-CODE: `# counters["arity"].append(len(products))`
3. RUNREACTANTS_EXCEPTION_RATE: Fraction of `_reduce` calls that
   raise `ReactionError` (RDKit parse failure). Unitless.
   Healthy: ≈ 0 for well-formed reactants.
   PSEUDO-CODE: `# wrap RunReactants: counters[rule]["err"] += 1`
4. MASS_BALANCE_PASS_RATE: `verify_mass_balance()` pass per rule.
   Unitless. Healthy: 1.0 (all click rules = empty stoichiometry).
   PSEUDO-CODE: `# run verify_mass_balance("CuAAC") at module load`
5. RATE_PREDICTOR_ATTACH_RATE: Fraction of rules with
   `rule.rate_predictor is not None` after
   `attach_all_rate_predictors()`. Unitless. Healthy: 1.0.
   PSEUDO-CODE: `# if rule.rate_predictor is None: counter["unfit"] += 1`
6. CATALYST_REQUIREMENT_COVERAGE: Fraction of rules whose
   `requires_catalyst` field is non-None when the rule's reaction
   pathway needs one (e.g. CuAAC requires Cu(I)). Unitless.
   Healthy: 1.0.
   PSEUDO-CODE: `# walk REACTION_RULES; check requires_catalyst`
7. THIOLENE_CAN_APPLY_RATE: Fraction of (thiol, alkene) pairs passing
   `ThiolEne.can_apply`. Unitless. Healthy: high for tile pairs that
   carry S-H and C=C.
   PSEUDO-CODE: `# counters["thiolene_ok"] += int(rule.can_apply(t,a))`

Currently tracked: N (no `history` field; `_rdkit_reaction` caches
compiled templates in-process).

Target values: stoichiometry = {} for all 5 rules (enforced by
`verify_mass_balance` at import time).

---

### Layer 5 — Rate Predictors (rate_predictor.py)

Data flow: Receives `(smiles_a, smiles_b)` pairs; produces predicted
isolated yield in [0,1] via `HeuristicRegressor` trained on
`LITERATURE_YIELDS` (6 reactions × 10 rows).

Metrics (5):

1. PREDICTOR_TRAIN_R2: `RatePredictor.train_r2` per reaction (already
   recorded on the dataclass). Unitless. Healthy: ≥ 0.6 on a hand-
   curated 10-row dataset.
   PSEUDO-CODE: `# already in `RatePredictor.train_r2` — log on .for_reaction()`
2. FEATURE_DIM_OK: Asserts `len(smiles_pair_features(a,b)) == 8`.
   Unitless. Healthy: True always.
   PSEUDO-CODE: `# if len(feats) != 8: counter["bad"] += 1`
3. PREDICTION_CLIP_RATE: Fraction of raw `model.predict` outputs
   outside [0,1] that get clipped. Unitless. Healthy: low (< 0.1);
   high clipping = poor extrapolation.
   PSEUDO-CODE: `# raw = model.predict(X); if raw<0 or raw>1: counter["clip"] += 1`
4. CITATION_COVERAGE: `len(RatePredictor.citations) / 10` per rule.
   Unitless. Healthy: 1.0 (10 literature yields per reaction).
   PSEUDO-CODE: `# if len(p.citations) != 10: counter["missing"] += 1`
5. BACKEND_USED: Per-rule `HeuristicRegressor.backend_` value (sklearn
   vs pysr). Categorical. Healthy: `sklearn` in ROCm Triton env
   (PySR fallback expected).
   PSEUDO-CODE: `# log priors[k].backend_`

Currently tracked: Y (partial — `train_r2` and `citations` exist per
`RatePredictor`; the module-level `RATE_PREDICTORS` dict caches fitted
models but no rolling history).

Target values: CuAAC median ≈ 0.90, SPAAC ≈ 0.86, SPC ≈ 0.89,
DielsAlder ≈ 0.89, ThiolEne ≈ 0.87 (per hand-curated median in
`LITERATURE_YIELDS`).

---

### Layer 6 — Tile Library (click_tiles.py + library.py)

Data flow: Receives SMILES + tag list (azide/alkyne/partner);
produces 12 `Tile` instances with 3D coords, descriptors, and
deterministic `tile_id`. `build_all_click_tiles()` returns the canonical
Phase-0 library.

Metrics (6):

1. TILE_COUNT_PER_GROUP: `len(AZIDE_TILES)`, `len(ALKYNE_TILES)`,
   `len(PARTNER_TILES)`. Unit: tiles. Healthy: 4+4+4=12.
   PSEUDO-CODE: `# assert len(STANDARD_12_TILES) == 12`
2. EMBED_SUCCESS_RATE: Fraction of tiles where
   `build_click_tile` produced a non-`None` `coords` tensor.
   Unitless. Healthy: ≈ 1.0 with `randomSeed=0xC11C`.
   PSEUDO-CODE: `# if tile.coords is None: counter["embed_fail"] += 1`
3. CANONICAL_SMILES_UNIQUE: `len(set(t.smiles for t in tiles))`.
   Unitless. Healthy: 12 (no duplicates).
   PSEUDO-CODE: `# assert uniqueness in `_ensure_built`
4. SAS_DISTRIBUTION: `tile.sas_score` mean/percentiles. Unit: SA score
   1–10. Healthy: 1.0–4.0 for hand-curated tiles.
   PSEUDO-CODE: `# append tile.sas_score to history`
5. MW_LOGP_TPSA_BOUNDS: min/max of MW, logP, TPSA across tiles.
   Unit: g/mol, unitless, Å². Healthy: MW ∈ [40, 200], logP ∈ [-1, 4],
   TPSA ∈ [0, 90] for the 12 reference tiles.
   PSEUDO-CODE: `# record (mw, logp, tpsa) per tile build`
6. TILE_ID_DETERMINISM: Two builds produce identical `tile_id` set
   (hash check). Unitless. Healthy: True.
   PSEUDO-CODE: `# first_run_ids = {t.tile_id for t in STANDARD_12_TILES}`

Currently tracked: N (no `history`; `_CACHE` global caches the list).

Target values: 12 tiles exactly, all embed successfully with the
fixed seed.

---

### Layer 7 — Type Predicates (predicates.py)

Data flow: Receives any ligand-like input (RDKit Mol,
`MoleculeClosedTerm`, `.rdkit_mol`); produces `True/False` per
`TypePredicate` plus `ill_typed_reasons`. Standard: `LIPINSKI`,
`VEBER`, `EGAN`, `REOS`.

Metrics (5):

1. PASS_RATE_PER_PREDICATE: Fraction of inputs satisfying each
   `TypePredicate` on a tile library / sampled drug-like set.
   Unitless. Healthy: LIPINSKI ≥ 0.7, VEBER ≥ 0.6, EGAN ≥ 0.5,
   REOS ≥ 0.4 (per typical ChEMBL distribution).
   PSEUDO-CODE: `# counters[p.name] += int(p(mol))`
2. DESCRIPTOR_COMPUTE_MS: Wall-time for `_descriptors(mol)` per call.
   Unit: ms/call. Healthy: < 5 ms on warm RDKit.
   PSEUDO-CODE: `# t0=time.perf_counter(); d=_descriptors(m); history.append(...)`
3. ILL_TYPED_REASON_FREQ: Histogram of which constraint fails first
   (`Lipinski > Veber > Egan > REOS`). Unit: counts. Healthy: Ro5
   dominates in early-stage libraries.
   PSEUDO-CODE: `# ill_typed_reasons(mol, ALL_ADMET) and tally[reason]`
4. RDKIT_DESCRIPTOR_MISS_RATE: Fraction of `_descriptors(mol)` calls
   that hit the `ValueError` branch (no RDKit Mol available).
   Unitless. Healthy: ≈ 0.
   PSEUDO-CODE: `# wrap _descriptors with try/except ValueError`
5. WELL_TYPED_FRACTION: `well_typed(mol, ALL_ADMET)` True rate across
   a batch. Unitless. Healthy: 0.3–0.5 over ChEMBL subsets.
   PSEUDO-CODE: `# counter["typed"] += int(well_typed(m, ALL_ADMET))`

Currently tracked: N (no `history`; results returned per call only).

Target values: 12 standard click tiles mostly satisfy LIPINSKI;
azide-rich tiles may trip HBD ≤ 5 depending on substituents.

---

### Layer 8 — Binding Types (binding/types.py)

Data flow: Receives `(ligand, BindingSite)`; produces
`BindingTypeCheckResult(success, pic50_estimate, violated_constraints,
details)`. Canonical sites: `MMP2_ACTIVE`, `PT_DNA_MAJOR_GROOVE`,
`KINASE_ATP`, `PROTEASE_GENERIC`.

Metrics (6):

1. TYPECHECK_SUCCESS_RATE: Fraction of ligands with
   `result.success == True` per site. Unitless. Healthy: site-
   dependent (MMP2 harder than generic protease).
   PSEUDO-CODE: `# counter[site.name]["ok"] += int(result.success)`
2. PIC50_DISTRIBUTION: mean / median / p10 / p90 of
   `result.pic50_estimate` per site. Unit: pIC50 (dimensionless, [0,12]).
   Healthy: median ≈ 6–8 on drug-like candidates for matching site.
   PSEUDO-CODE: `# append result.pic50_estimate per typecheck`
3. CONSTRAINT_FIRST_FAILURE: Histogram of which constraint fails first
   per site. Unit: counts. Healthy: site-specific (MMP2:
   hydroxamic_acid_zbg dominates).
   PSEUDO-CODE: `# if result.violated_constraints: tally[result.violated_constraints[0]] += 1`
4. GEOM_BETA_PASS_RATE: Fraction where `details['geometric_check']`
   reports OK. Unitless. Healthy: high for site-matched chemotypes.
   PSEUDO-CODE: `# counter["geom_ok"] += int(not result.violated_constraints or 'geom' not in str(result.violated_constraints))`
5. WARHEAD_HIT_RATE: Fraction of ligands where
   `hydroxamic_acid_present(mol) == True` (or other warhead functions).
   Unitless. Healthy: site-specific — high for focused libraries.
   PSEUDO-CODE: `# counter["warhead"] += int(hydroxamic_acid_present(mol))`
6. PIC50_COMPONENT_RESIDUAL: Stdev of
   `details['pic50_components']['final']` per site. Unit: pIC50.
   Healthy: σ < 2 indicates heuristic not bimodal.
   PSEUDO-CODE: `# collect pic50_components['final'] into per-site buffer`

Currently tracked: N (no `history`; full diagnostic bag lives in
`result.details` per call).

Target values: MMP2 site requires hydroxamic-acid ZBG → typical
ChEMBL MMP2 actives pic50 ≥ 8.

---

### Layer 9 — Search & Closed Loop
(search_alg/proof_search.py + pipeline/closed_loop.py)

Data flow: Receives `MoleculeClosedTerm` seed + `tile_library` +
`rules` + `target_predicates` + `binding_site`; produces top-K
candidates plus per-iteration `history` dict and
`_IterationRecord`-derived dicts.

Metrics (8):

1. BEST_SCORE_TRAJECTORY: `best_score` per iteration (already recorded
   in `MCTSProofSearch.history[i]['best_score']`). Unitless. Healthy:
   monotonically non-decreasing for a converged search.
   PSEUDO-CODE: `# already in history; chart over iterations`
2. N_STATES_EXPLORED: Tree size after each iteration (already in
   history). Unit: nodes. Healthy: grows sub-linearly with early
   iterations.
   PSEUDO-CODE: `# already in history['n_states_explored']`
3. N_SATISFYING: Number of leaves that satisfy predicates AND bind
   site (already in history). Unit: count. Healthy: ≥ 1 by end of run.
   PSEUDO-CODE: `# already in history['n_satisfying']`
4. PUCT_EXPLOIT_RATIO: Fraction of `_select_child` calls that pick
   the child with max `Q` (vs max `U`). Unitless. Healthy: increases
   with iterations (search converges).
   PSEUDO-CODE: `# in _select_child: if best_score == Q: counter["exploit"] += 1`
5. ROLLOUT_GUIDED_RATIO: Fraction of rollout picks using the
   `_rollout_pick_guided` path. Unitless. Healthy: ≈ 1.0 when prior
   is fitted and `rollout_epsilon > 0`.
   PSEUDO-CODE: `# if use_guided and rng.random() >= eps: counter["guided"] += 1`
6. DIRICHLET_APPLIED: Boolean flag set in
   `_apply_dirichlet_to_root` (already a local var). Unitless.
   Healthy: True once per search when alpha > 0.
   PSEUDO-CODE: `# if root_dirichlet_applied: counter["dirichlet"] += 1`
7. CLOSED_LOOP_ITERATION_LATENCY: Wall-time per
   `LamClickDesignLoop.run()` iteration. Unit: seconds.
   Healthy: < 60 s for 10 simulations × depth 3.
   PSEUDO-CODE: `# t0=time.time(); self._iter(...); history.append(time.time()-t0)`
8. EQUATION_CHANGE_RATE: Fraction of iterations where
   `extracted_formula != prev_iter_formula`. Unitless. Healthy:
   decreases as loop converges.
   PSEUDO-CODE: `# if formula != self.paper_equation: counter["changed"] += 1`

Currently tracked: Y — `MCTSProofSearch.history` (per-iter dict with
`iteration, best_score, mean_score, n_states_explored, n_leaves,
n_satisfying, best_state`) and `LamClickDesignLoop.history`
(`_IterationRecord` with `iteration, best_smiles, best_lambda_expr,
best_score, extracted_formula, top_k_smiles, top_k_scores`).

Target values: best_score ≥ 0.5 by iteration 3; n_satisfying > 0 in
standard Lambda demo runs.

---

### 10. Cross-Layer Metrics

Composition metrics that need data from ≥ 2 layers:

- `synthesis_success = n_satisfying / n_sims` — divides Layer 9
  (`MCTSProofSearch.history[i]['n_satisfying']`) by `n_simulations`,
  crossing Layers 9, 7 (predicates), 8 (binding) and 4 (reactions).
  Healthy: ≥ 0.01 at 1000 sims.
- `SA_score_mean = Ertl(scored_smi)` — average synthetic
  accessibility over top-K candidates (Layer 9's
  `top_k_smiles`), composes Layers 9, 6, and the SA scorer.
  Healthy: ≤ 4.0.
- `retrosynth_feasibility = mean(r_retro(state) for state in top_k)`
  — uses Layer 9's top-K crossing Layers 9, 6, 4, and the
  retrosynthesis adapter. Healthy: ≥ 0.6.
- `binder_pass_rate = typecheck_ok / total` — Layer 8 success rate
  on Layer 9 candidates. Healthy: site-dependent.
- `mass_balance_violations = sum(rule.stoichiometry != {} for rule
  in REACTION_RULES)` — Layer 4 invariant, should be 0 always.
- `pipeline_throughput = n_candidates / closed_loop_wall_time` —
  Layer 9 throughput. Healthy: ≥ 1 candidate/s for default config.
- `end_to_end_yield_proxy = mean(rule.predict_yield(a,b) for top_k)`
  — combines Layer 5 predictions with Layer 9 top-K; cross-layer
  measure of click-reaction success. Healthy: ≥ 0.8.

---

File: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_layer_metrics.md`

Per-layer metric counts: L1=5, L2=6, L3=6, L4=7, L5=5, L6=6, L7=5,
L8=6, L9=8. Total: 54 metrics across 9 layers + 7 cross-layer
metrics. Word count of metric catalogue body (excluding headings and
file path): ≈ 940.
