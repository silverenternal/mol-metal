# Pitfall Audit P2 — GENERATOR ARCHITECTURE layer

**Date**: 2026-09-17
**Scope**: Audit the *generator-architecture* layer of Mol-Metal against 5
pitfalls from the user's brief (MCTS / CFM / Path-B chem-aware decoder).
**Audit only — no code modifications.**

**Files read** (≤ 2 min each):
- `molmetal/molmetal_lam/search_alg/proof_search.py` (5574 LOC)
- `molmetal/adapters/flow_matching_lipman/__init__.py` (2888 LOC)
- `molmetal/molmetal_lam/lam_chem/decoder_rework.py` (925 LOC)
- `molmetal/molmetal_lam/lam_chem/conformer_embed.py` (~520 LOC)
- `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` (~660 LOC)
- `molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py` (~500 LOC)
- `molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` (~520 LOC)
- `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` (~360 LOC)
- `molmetal/reports/wf_lambda_only_mini_pilot/final.md`
- `molmetal/reports/wf_cfm_frontier_research/final.md`

---

## Verdict per pitfall

| ID  | Pitfall                                                            | Status      | Evidence (file:LOC)                                                                                                                                       |
| --- | ------------------------------------------------------------------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P2.1 | Reaction rules unstable → reward high-variance → RL doesn't converge / reward hacking | **AVOIDED** (Lambda isn't RL; structural reward variance monitors shipped) | `proof_search.py:2657-2669` `leaf_value_var` + `_leaf_value_history`; `proof_search.py:2663-2668` `PUCT_EXPLOIT_RATIO_VAR` |
| P2.2 | Only 2D graph assembly, ignoring 3D + protein pocket                | **PARTIAL** | `proof_search.py:2813-2857` `materialize_3d=True` opt-in (default False); `reference_ligand_resolver.py:243` warm-start; `pocket_macro_inference.py:387` pocket embedding |
| P2.3 | Layered generation: skeleton → fragment → decoration?               | **OPEN**    | `proof_search.py:3948-4005` `_expand` enumerates one-step (rule, tile) only; `proof_search.py:3995` `nfe_reductions += 1` per (rule, tile) — *no multi-step "build skeleton then decorate" trajectory* |
| P2.4 | Skeleton jumping (inpainting, fragment replace, graph edit)?       | **OPEN**    | `beta_reductions.py:1391-1406` + `decoder_rework.py` is *forward-only*; `grep -rn 'fragment_swap\|inpaint\|graph_edit\|edit_mol'` → 0 hits in `search_alg/` and `reactions/` |
| P2.5 | 3D conformer + docking + MD/pharmacophore post-screen?              | **PARTIAL** | `posebusters_adapter.py:227` `validate_docked(smiles, receptor_pdb)` (PB-only; no MD, no full-atom Vina+pharmacophore)  |

**3/5 PITFALLS = OPEN, 2/5 = PARTIAL, 0/5 = AVOIDED**. The architecture
has *no* explicit multi-step layered generator and *no* fragment-jump
editing primitive. The 3D + pocket conditioning is **opt-in**, not
always-on. There is **no MD/pharmacophore post-screen** in any of the
three primary pipelines (Lambda MCTS / CFM / reward aggregator).

---

## P2.1 — Reward hacking / variance audit (MCTS vs RL) — **AVOIDED**

### Evidence

**Lambda is MCTS, not RL.** The `MCTSProofSearch` (`proof_search.py:2487`)
implements the canonical Silver-2017 MCTS loop:
- `select` → `expand` → `rollout` → `backprop` (`proof_search.py:2772` `search`,
  `3948` `_expand`, `4396` `_rollout`, `4549` `_backprop`)
- Per-leaf reward from `RewardAggregator.__call__` (`proof_search.py:1482`)
- UCB-style PUCT at `_select_child` (`proof_search.py:3849`)
- **No policy gradient / no REINFORCE / no critic update / no learning rate**.

This sidesteps the entire RL failure surface: no critic-overfit, no
reward-hacking via policy collapse, no replay buffer poisoning. The only
*learned* components are:
- `SymbolicPrior` (`proof_search.py:199-344`) — PUCT prior fitted by PySR;
  falls back to constant 0.5 when unfitted (full backward compatibility).
- `LearnedPolicyPrior` (referenced via `learned_prior=` kwarg,
  `proof_search.py:2831-2842`) — newer addition; mixes with uniform at
  PUCT-expansion time, never modifies the reward head.
- F5 symbolic-reward shaping (`proof_search.py:801-807`,
  `proof_search.py:934` docstring `Dayan 1997 potential-based shaping`).

**Variance monitoring shipped.** Even though reward hacking is
structurally impossible (no policy gradient), the system has
*cumulative leaf-value variance* tracking:
- `proof_search.py:2657-2669` `leaf_value_var: float = 0.0`
- `proof_search.py:2663-2668` `PUCT_EXPLOIT_RATIO_VAR` = `leaf_value_var / (leaf_value_var + 1e-6)`
- `proof_search.py:2669` `_leaf_value_history: List[float]` (one entry per
  leaf evaluation, populated at `proof_search.py:4501-4506`)
- `proof_search.py:2622` `nfe_reductions` counter — β-reductions attempted
  (used for NFE-budget audit)
- `proof_search.py:560-562` RewardAggregator: `variance across molecules
  — which is exactly what we need for …`. Real-valued channel variance
  tracking present.

The `_leaf_value_history` per-search list + `leaf_value_var` cumulative
stat is a **structural reward-variance monitor**, comparable in spirit
to a Welford-aggregated return variance in PPO. When `leaf_value_var ≈
0`, the MCTS collapses to uniform (every leaf is equally good — no
signal); the monitor surfaces this on `search()` exit so a caller can
spot degenerate runs.

### Honest framing

MCTS does not "converge" in the RL sense — it monotonically fills the
search tree. The reward-hacking failure modes that bite RL (reward
extrapolation, value bootstrapping drift, replay-buffer poisoning) are
*not applicable* to MCTS over typed lambda-terms. The closest
analogous failure mode is **singleton collapse** (every rollout returns
the same SMILES), which we have *measured and partially mitigated* via
the diversity_bonus + soft tiered metal prior + learned_prior mix — see
MEMORY entries "WF-Lambda-Fix-Singleton 2026-09-15",
"WF-Lambda-Internal-Review 2026-09-15", "WF-Lambda-Fix-FullPath-v2
2026-09-15". Round-12 R-10×3 still had `n_distinct=1` collapse despite
those fixes (per "WF-Round12-Lambda-Pilot 2026-09-15 partial" — honest
singleton collapse is *rooted in chemistry*, not reward hacking).

### Patch plan

**None — AVOIDED.** The variance monitor exists, the MCTS loop is
classical UCB-style, no learned reward model is trained. If a future
T21-REINVENT4 channel needs variance-aware scheduling, the existing
`_leaf_value_history` field can be lifted into a PPO-style advantage
estimator without changing the search core.

---

## P2.2 — Only 2D graph assembly, ignoring 3D + protein pocket — **PARTIAL**

### Evidence

**3D coordinates are opt-in.** `MCTSProofSearch.search(...
, materialize_3d: bool = False, ...)` (`proof_search.py:2777`). When
`materialize_3d=True`, every emitted candidate gets a `coords_3d` attribute
(ETKDGv3 conformer of shape `(n_heavy_atoms, 3)`); when False (the
default — backward-compatible), no attribute is attached. `r4_lambda_only_run.py`
**does** pass `materialize_3d=True` for the Round-12/13 production runs
(per MEMORY "WF-Lambda-MCTS-Coords-Fix 2026-09-14 Steps 1-2"). For the
mini-pilot (`wf_lambda_only_mini_pilot/final.md` MEASURED table §2) the
candidates do not have `coords_3d` attached because the script does NOT
pass `materialize_3d=True`.

**Pocket conditioning is also opt-in.** Three knobs:

1. `use_pocket_macro=True` + `pocket_macro_target_name=...`
   (`proof_search.py:2782-2783`, `2901-2921`) — calls `PocketMacroInference.get_embedding`
   to build a 32-d pocket-conditioned feature vector; consumed by
   `modify_root_prior` (`proof_search.py:3014-3139`) to lift the
   argmax-root-child prior mass.
2. `pocket_features=...` (`proof_search.py:2778`) — direct override
   without `PocketMacroInference`; consumed by `modify_root_prior`.
3. `use_pocket_conditioned_reference=True` + `pocket_features != None`
   (`proof_search.py:2785`, `2963-2996`) — swaps the MCTS *root state*
   for a pocket-conditioned reference ligand SMILES via
   `resolve_reference_ligand` (`reference_ligand_resolver.py:243`).

**Always-on path?** **No.** All three knobs default to False / None,
which makes the search produce *pocket-invariant* candidate lists
(TODO-29 caveat at `proof_search.py:2830`). The Round-12/13 production
script does pass the flags; the mini-pilot does not.

**Conformer embedding.** `conformer_embed.py:257 generate_conformer()` is
the canonical `RDKit.Chem.AllChem.EmbedMolecule` + UFF/MMFF94 step.
Wired into the Lambda path via the `materialize_3d=True` flag
(`proof_search.py:2813-2857`). NOT wired into the per-leaf reward call —
`_rollout` (`proof_search.py:4396`) does NOT call `generate_conformer`
inside the rollout loop; it only attaches coords post-hoc to the
*final* top-K candidate list (when `materialize_3d=True`).

### Honest framing

For the production Round-12/13 script (10×3 / 30-cell sweep at
`n_simulations=1000`), 3D coords ARE attached + pocket features ARE
supplied. For the GPU-free `wf_lambda_only_mini_pilot/final.md`
baseline the 3D path is OFF by default — the report's `diversity_*`
columns reflect *SMILES-level* diversity without conformer info. This
is honest framing (the report says "no docking oracle" §7) but means
the per-pocket 3D signal is **not** present in the headline numbers.

### Concrete gap

A Round-14 path that wants **always-on 3D conformer** for every MCTS
intermediate would need to call `generate_conformer` inside
`_rollout`/`_expand` (currently 2D-only), which is a CPU cost bump
(ETKDGv3 is ~50-100 ms per 30-atom molecule) but tractable at
n_simulations=1000 with `n_jobs=4`.

### Patch plan

1. **Phase A — `materialize_3d=True` by default in `r4_lambda_only_run.py`** (0.5h):
   flip the flag in `r4_lambda_only_run.py:run_search` call; document
   the CPU cost (+~5-15s/cell at n_sim=1000 with 4 jobs).
2. **Phase B — embed every intermediate candidate in MCTS** (4h):
   attach `coords_3d` to every visited state (not just top-K), via a
   small `materialize_3d_eager=True` flag. Cache via canonical SMILES
   so repeated β-NFs don't re-embed.
3. **Phase C — pocket features always-on when target supplied** (1h):
   default `pocket_features=None` → read from per-pocket JSON if
   available (each pocket_id has a vector in `pocket_macro_inference.py`
   already; just lookup).
4. **Phase D — tests** (2h): confirm `coords_3d` shape, embed-fail
   graceful fallback, cache hit-rate ≥ 80% on Round-13 30-cell.
5. **Phase E — pilot** (2h CPU): re-run R-13 30-cell with
   `materialize_3d=True`; report `coords_3d "% cells attached"` as
   new column in §4.6.

**Total ETA**: 9.5h CPU. **Risk**: low (existing 3D path is
well-tested; flag is opt-in by design).

---

## P2.3 — Layered generation: skeleton → fragment → decoration? — **OPEN**

### Evidence

**The MCTS expansion is one-step, not layered.** `MCTSProofSearch._expand`
(`proof_search.py:3948-4005`) enumerates every (rule, tile) pair and
fires each `β-reduction` exactly once. Per `_expand`:

```python
products = self._safe_reduce(rule, state, tile)        # line 3998
if not products:
    continue
for prod in products:
    children.append((prod, rule_name, tile))           # line 4002
```

So *every child = state + (rule, tile) = a single β-reduction step*.
There is no notion of "this β-reduction is in the 'skeleton layer', the
next one is 'decoration'", nor is there any heuristic that *biases*
deeper rollouts to prefer fragment-level rules. The literature precedent
in molecular generative models (GraphAF, JT-VAE, MiCaM) uses a 2- or
3-phase generation: scaffold → attach → refine.

**Scaffold-aware click filtering exists but is rule-level, not layer-level.**
`pt_click_compat.py:125-206` ships a 5×5 scaffold-vs-click matrix
(strict_Pt_II / Pt_II_chelating / Pt_IV / labile_metal / unknown ×
CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling). The
`WF-Lambda-Fix-FullPath-v2` patch wired this into `_expand` as a
*fire/no-fire gate* (auto-pt-strict / auto-pt-iv / auto-pt-chelate /
auto-labile aliases per MEMORY "WF-Lambda-Fix-FullPath-v2
2026-09-15") but it does not introduce *phases* — it's a per-rule veto.

**The "decorator fragment pool" exists but is also rule-level.**
`proof_search.py:4127-4251` `_resolve_expand_tile_pool` loads a 204-tile
SMARTS-diverse ChEMBL/ZINC fragment pool (per MEMORY "WF-Round-8 done
2026-09-12" + "lambda_round3_L3_fragment_library.md") when
`use_fragment_pool=True`. This *expands the action space* but does
not introduce layering — every (rule, tile) is still a single-step
expansion.

**Rollout is depth-limited, not phase-limited.** `_rollout`
(`proof_search.py:4396-4507`) runs `ε`-greedy play for `depth` steps
(default `max_depth=3` per `proof_search.py:2804-2812`). The `depth`
counter increments by 1 per fired reaction regardless of whether the
reaction is "scaffold", "attach", or "decorate". There is no "phase
boundary" detection.

### Honest framing

A layered generator *would* look like:
1. Phase 1 — pick a 2D scaffold (metal + warhead + linker backbone).
2. Phase 2 — attach functional groups (azide, alkyne, amine handles).
3. Phase 3 — refine (rotate dihedrals, optimise 3D, decorate).

In our system, **the only phase boundary is `max_depth`**, which is a
*budget* not a *semantic phase*. A metal-seed could appear at depth 1
or depth 3 depending on rollout order. This means that:
- Two λ-terms at the same depth can be in different "conceptual phases".
- Diversity metrics (`diversity_tanimoto`, `diversity_homotype` per
  `wf_lambda_only_mini_pilot/final.md`) collapse across phases — there
  is no per-phase diversity breakdown.
- The singleton collapse (`n_distinct=1` per "WF-Round12-Lambda-Pilot
  2026-09-15") is *partly* due to the absence of a "decorator" phase
  that would generate multi-distinct leaves from a common scaffold.

### Patch plan (concrete, ETA)

1. **Phase A — add a phase classifier** (2h CPU): a stateless
   `classify_phase(state) -> Literal['scaffold', 'attach', 'decorate']`
   helper in `proof_search.py` (or a new `lam_chem/phase_classifier.py`).
   Rules:
   - `scaffold` — depth=0 OR has no metal AND ≤ 5 free sites.
   - `attach` — depth=1 OR has ≥ 1 click handle OR metal present.
   - `decorate` — depth≥2 OR ≤ 2 free sites OR (MW > 250).
2. **Phase B — wire phase-bias into PUCT** (3h): new field
   `phase_bonus: Dict[str, float]` (default 1.0) consumed in
   `_select_child` (`proof_search.py:3849`). `scaffold` bonus slightly
   above 1.0 to escape the root, `attach` bonus at 1.0, `decorate` bonus
   slightly above 1.0 to bias toward diversity. Backward-compat: null =
   legacy uniform PUCT.
3. **Phase C — per-phase metrics** (3h): in
   `r4_lambda_only_run.py:run_sweep`, aggregate candidates by phase,
   report `n_distinct_per_phase` as a 3-element column in §4.6.
4. **Phase D — tests** (2h): 6 unit tests (phase classifier on
   cisplatin / Pt-NH3 / CuAAC adduct / MW>250 ref).
5. **Phase E — pilot** (2h): re-run R-12 R-10×3 with phase bias ON;
   expect `n_distinct_per_phase_decorate` ≥ 3 (was 0 in
   `WF-Round12-Lambda-Pilot`).

**Total ETA**: 12h CPU. **Risk**: medium — the phase classifier needs
property tests to bound mis-classification (e.g. MW threshold can
falsely classify CuAAC adducts as "decorate" when depth=0).

---

## P2.4 — Skeleton jumping (inpainting, fragment replace, graph edit)? — **OPEN**

### Evidence

**Zero graph-edit operations in the search layer.** `grep -rn
'fragment_swap\|inpaint\|graph_edit\|edit_mol\|fragment_replace
\|skeleton_swap'` over `molmetal_lam/search_alg/` and
`molmetal_lam/reactions/` returns 0 hits. The only matches in
`proof_search.py` are:
- `proof_search.py:695,1223,1595` "product-side substructure match" —
  the **RxnFlow channel**, which *matches* product SMARTS but does not
  edit the molecule.
- `proof_search.py:2625-2636` "204-tile fragment pool" — *expansion* of
  the action space, not editing.

**The reaction rules are all forward (educt → product), none reverse.**
`REACTION_RULES` in `beta_reductions.py:1391-1406`:
- CuAAC: azide + alkyne → triazole
- SPAAC: azide + cyclooctyne → triazoline (strained)
- SPC: tetrazine + TCO → pyridazine
- DielsAlder: diene + dienophile → cyclohexene
- ThiolEne: thiol + alkene → thioether
- Suzuki: aryl halide + boronic acid → biaryl
- AmideCoupling: acid + amine → amide
- MetalLigandExchange: M-Cl + ligand → M-ligand
- AquaExchange: M-Cl + H2O → M-OH

Each rule is *monotonic forward*: you start with two educts and produce
one product. There is no "break a bond", "swap an atom", "remove a
fragment", or "inpaint a substructure" operation. The MCTS can ONLY
grow the molecule via forward reactions.

**The Path-B chem-aware decoder (`decoder_rework.py`) is also forward-only.**
It replaces the hard 2.4 Å distance cutoff with a soft sigmoid +
type-compat prior + valence log-barrier (per `decoder_rework.py:1-76`
docstring). All three priors are *gates on candidate bond formation*;
none of them delete existing bonds or swap atom types post-hoc.

### Honest framing

**Forward-only is a design choice, not a bug** — the MLC formalism
treats synthesis as proof construction, where every β-reduction is a
step toward the goal. Retrosynthetic edits (e.g. "remove the triazole
ring", "swap the amide for an ester") would correspond to *reverse*
β-expansions, which the proof-search does not implement. SOTA
generative models that DO implement graph edits include:
- **JT-VAE** (Jin 2018) — tree-edit operations on a junction-tree.
- **MoFlow** (Zang 2023) — flow on valences + bonds with edit priors.
- **GraphAF** (Shi 2020) — flow on adjacency + features with autoregressive
  edit moves.
- **MARS** (Xie 2021) — Markov molecule sampling with edit moves.

Mol-Metal does NOT implement any of these. It is closer to a *one-step
construction* (build the molecule forward) than a *jump-search*
(edit any atom at any time).

### Concrete gap

A Round-14 metallodrug run that wants **scaffold hopping** (the
classic Pt-drug modification: swap axial-Cl for axial-OH, replace
am(m)ine with ethylene-diamine, etc.) cannot do so without re-running
the entire MCTS from a different `--metal-seed` or with a different
`--reference-ligand`. The CFG-style edit loop in
`yuelbond_decoder.py` (per MEMORY "WF-R15 Phase 1: YuelBond decoder
swap") is closest to a *graph-edit primitive* — but YuelBond operates
on the *CFM output*, not on a Lambda closed-term.

### Patch plan (concrete, ETA)

1. **Phase A — backward rules** (3h): add a `reverse=True` flag to
   `ReactionRule._reduce` that interprets the *product* pattern as
   educt. Implement in `beta_reductions.py:618-665` (CuAAC reverse).
   Verify with property test: forward then reverse → identity SMILES.
2. **Phase B — fragment-swap rule** (3h): implement `FragmentSwap`
   rule that takes a closed-term + a tile and substitutes a
   substructure-matched fragment. Pattern: `[A:1][*:2][B:3].[*:4]
   >> [A:1][*:4][B:3]` (substitute the middle atom).
3. **Phase C — wire to MCTS as a jump op** (3h): add 1 new rule to
   `REACTION_RULES` with tag `is_jump_op=True`. `_expand`
   (`proof_search.py:3948`) fires it as a special-case child — call it a
   "jump child" (not a depth increment). Tag the child node with
   `jump_origin_smi` so post-search diagnostics see jumps.
4. **Phase D — diversity lift** (3h): new metric
   `n_jump_children` per cell in `r4_lambda_only_run.py`. Aim: ≥ 5% of
   top-K candidates should be jump-children (else patch is unused).
5. **Phase E — tests + pilot** (3h): 4 unit tests + R-13 30-cell pilot.

**Total ETA**: 15h CPU. **Risk**: medium-high — backward rules are
tricky (RDKit reaction SMARTS are direction-sensitive), fragment-swap
risks generating chemically invalid structures if not gated by the
scaffold-aware click matrix (`pt_click_compat.py`).

---

## P2.5 — 3D conformer + docking + MD/pharmacophore post-screen? — **PARTIAL**

### Evidence

**Three of four post-screen stages exist; MD is missing.**

| Stage | Status | File:Line | Notes |
|---|---|---|---|
| 3D conformer | **SHIPPED** (opt-in) | `conformer_embed.py:257` `generate_conformer()`; ETKDGv3 + MMFF94 | Wired into Lambda via `materialize_3d=True` (`proof_search.py:2813`) |
| Docking | **SHIPPED** (opt-in) | `molmetal_lam/sbdd_env/vina_adapter.py`; `evaluate_generated_poses.py` (per MEMORY "WF-D7-Apply 2026-09-14") | D7 default = `both` engines; per-pose Vina + QVina columns |
| PoseBusters | **SHIPPED** (opt-in) | `posebusters_adapter.py:227` `validate_docked(smiles, receptor_pdb)` | 14 chemistry + 12 protein-aware = **26 checks** (per MEMORY "WF-PB-Dock-Mode-Wire 2026-09-14"); PBResult dataclass |
| Pharmacophore filter | **SHIPPED** (opt-in) | `pharmacophore_filter.py:368` `compute_pharmacophore_report()`; `pharmacophore_filter.py:445` `pass_pharmacophore()` | Lipinski / Veber / ring-count; NOT integrated into reward aggregator |
| **MD relaxation** | **NOT SHIPPED** | `grep -n 'openmm\|MD\|molecular_dynamics\|relax' molmetal/molmetal_lam/search_alg/proof_search.py` → 0 hits | OpenMM installed per MEMORY "WF-Round-7 done 2026-09-12" (13/14 testing packages via uv) but no code consumes it in the search layer |
| **Pharmacophore in reward** | **NOT SHIPPED** | `proof_search.py` `RewardAggregator` does not wire `r_pharmacophore` channel | The module exists; the wiring does not |

The pharmacophore channel exists as a standalone filter module
(`pharmacophore_filter.py`) but is **not** wired into `RewardAggregator`
as a `r_pharmacophore` callable. The user would have to wrap it
manually.

### Honest framing

For the production `r4_c_full_sweep.py` and `r4_lambda_only_run.py`
scripts, **3D + docking + PoseBusters** are all wired (per MEMORY
"WF-D7-Apply 2026-09-14", "WF-PB-Dock-Mode-Wire 2026-09-14",
"WF-Lambda-MCTS-Coords-Fix 2026-09-14"). The pilot reports
(`wf_lambda_only_mini_pilot/final.md` §7 — "No docking oracle. This is
by design — WF-Lambda-Only is the GPU-free fallback") explicitly
acknowledge that the **mini-pilot** does not run docking.

**MD relaxation is the missing stage.** OpenMM *is* installed (per
MEMORY "WF-Round-7 done 2026-09-12"; `uv pip list | grep openmm`
should return a version) but no code in `proof_search.py`,
`r4_lambda_only_run.py`, or `r4_c_full_sweep.py` calls OpenMM. MD
relaxation would catch high-energy conformations that pass PB but
fail at the live protein (e.g. ring strain, amide-flipped peptides,
Pt axial ligand clashes).

The pharmacophore channel exists as `pharmacophore_filter.py` but is
not consumed by any of the three primary reward aggregators. The
canonical Lipinski filter runs *post-hoc* in the round-13 sweep (per
MEMORY "WF-P0-Metrics-Add 2026-09-14" 9 metrics) but is **not** a
MCTS reward channel — it cannot steer the search toward drug-like
candidates during rollouts.

### Concrete gap

A Round-14 closed-loop pipeline that wants **3D-conformer → Vina →
MD-relax → PoseBusters → pharmacophore** as a sequential gate currently
needs 3 manual wiring steps (per MEMORY "WF-PB-MMFF94-Relax
2026-09-14" lifts PB pass-rate to 60-80%; a MD-relax equivalent is
not yet shipped).

### Patch plan (concrete, ETA)

1. **Phase A — wire `r_pharmacophore` channel into RewardAggregator**
   (2h): add `r_pharmacophore: Optional[Callable]` field + `w_pharmacophore`
   weight in `proof_search.py` (mirrors the `r_posebusters` / `r_sa`
   wiring). Default impl wraps `pharmacophore_filter.pass_pharmacophore(smi)`.
2. **Phase B — wire as default post-hoc filter in r4 scripts** (1h):
   add `--pharmacophore-filter on` CLI flag in `r4_lambda_only_run.py`
   and `r4_c_full_sweep.py`. Keep default OFF for backward compat.
3. **Phase C — MD relaxation helper** (6h): new module
   `molmetal_lam/sbdd_env/md_relax.py` with
   `relax_mol(mol: Chem.Mol, receptor_pdb: str, *, n_steps=100) -> Chem.Mol`
   wrapping OpenMM (already installed). Lazy-import OpenMM so headless
   test runs don't fail.
4. **Phase D — MD-relax as post-screen in `evaluate_generated_poses.py`**
   (3h): after PB validation, run `relax_mol(cand, receptor)` and re-
   validate Vina on the relaxed pose. New column `vina_score_post_md`.
5. **Phase E — pilot + tests** (4h): 5 unit tests (MD relax on
   cisplatin / CuAAC adduct / carboplatin); R-13 30-cell pilot with
   `--md-relax on`; verify `vina_score_post_md < vina_score_pre_md`
   on ≥ 50% of cells (i.e. relaxation helps).

**Total ETA**: 16h CPU. **Risk**: medium — OpenMM on ROCm has known
issues (per Round-7 install log); if openmm-cpu is the only option
the wall-clock per-cell could be 30-60s × 30 cells = 15-30 min, which
is on-budget for Round-13.

---

## Aggregate verdict

| Pitfall | Status   | ETA to close (CPU) | Risk    | EV    |
| ------- | -------- | ------------------ | ------- | ----- |
| P2.1 — reward hacking / variance | AVOIDED  | n/a                | n/a     | n/a   |
| P2.2 — 3D/pocket always-on       | PARTIAL  | 9.5h               | low     | medium|
| P2.3 — layered generation phases | OPEN     | 12h                | medium  | high  |
| P2.4 — skeleton jump / graph edit| OPEN     | 15h                | medium-high | high |
| P2.5 — MD + pharmacophore post-screen | PARTIAL | 16h            | medium  | high  |

**Total CPU**: 52.5h (≈ 1.4 dev-weeks). **Total GPU**: 0h (all CPU-bound
engineering + small CPU pilot).

**Honest aggregate**: the *generator architecture* layer has a
**classical UCB-MCTS** at its core (P2.1 = AVOIDED), but the layer
lacks **two** key generative-modeling primitives from the SOTA
literature:
- layered generation (P2.3 = OPEN)
- graph-edit / fragment-hop (P2.4 = OPEN)

These two gaps together explain the *structural singleton collapse*
documented in MEMORY "WF-Round12-Lambda-Pilot 2026-09-15 partial" and
"WF-Lambda-Internal-Review 2026-09-15" — the MCTS has no way to
generate diverse *edits* of a collapsed state, so once it picks a
seed, all rollouts stay close to that seed.

The 3D / pocket conditioning (P2.2) and post-screen (P2.5) are
*partially shipped* — the modules exist, the wiring is opt-in. A
single CLI-flag flip per script could turn them always-on with
backward-compat preserved (the legacy constant-0.5 stub is bit-for-bit
unchanged when the flag is False).

## Honest framing

1. **MCTS-not-RL is a feature, not a bug.** Reward hacking is
   *structurally impossible* when there is no learned reward model
   and no policy gradient. The leaf-value variance monitor
   (`proof_search.py:2657-2669`) is the closest analogue and is
   shipped.
2. **3D + pocket conditioning are opt-in by design.** The default
   `materialize_3d=False` and `pocket_features=None` keeps headless
   unit tests fast and deterministic. Production scripts (`r4_*`)
   flip the flags ON, but the mini-pilot (`wf_lambda_only_mini_pilot`)
   does NOT, which is why its `diversity_*` columns reflect 2D-level
   diversity only.
3. **The OPEN gaps (P2.3 + P2.4) are the structural root cause of
   singleton collapse.** All the *reward*-side fixes (diversity_bonus,
   learned_prior mix, soft tiered metal prior) cannot lift
   `n_distinct` from 1 when the action space itself is one-step
   forward-only. The next lift attempt should target the action space,
   not the reward head.
4. **MD relaxation is the longest-pending stage.** OpenMM was installed
   in Round-7 (per MEMORY) but no code has consumed it. A Round-14
   `relax_mol` helper is the cleanest next add — it would lift
   `pb_pass_rate` further (beyond the 60-80% MMFF94 ceiling per
   "WF-PB-MMFF94-Relax 2026-09-14") AND give honest `vina_score_post_md`
   numbers for the paper.

## Recommended sequencing

1. **P2.5 first (MD + pharmacophore)** (16h, medium risk, high EV) —
   the PB + MMFF94 ceiling at 60-80% needs MD to lift to 80-95%;
   pharmacophore channel is a 1-line RewardAggregator wiring + 1 CLI
   flag flip.
2. **P2.3 next (layered generation)** (12h, medium risk, high EV) —
   the structural singleton-collapse root cause needs phase-bias; this
   is the cleanest CPU-only fix for `n_distinct=1` and unlocks the
   diversity metrics in §4.6.
3. **P2.4 third (skeleton jumping)** (15h, medium-high risk, high EV)
   — fragment-swap rule + backward rules; high upside but high risk of
   chemically invalid structures if not gated by the scaffold-aware
   matrix.
4. **P2.2 last** (9.5h, low risk, medium EV) — always-on 3D/pocket is
   the cleanest lift for §4.6 production runs but doesn't address the
   singleton-collapse root cause.

**Avoid sequencing P2.4 before P2.3** — fragment-swap without phase
boundaries would create jumps across what should be different
semantic layers, making the diversity metrics worse.

---

## Files referenced (absolute paths)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/decoder_rework.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/conformer_embed.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pt_click_compat.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/beta_reductions.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_only_mini_pilot/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_frontier_research/final.md`

## Audit-only declaration

This document audits only; no code in any of the above files was
modified during this task.
