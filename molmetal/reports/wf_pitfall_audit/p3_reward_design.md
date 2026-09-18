# Pitfall Audit P3 — REWARD DESIGN layer

**Date**: 2026-09-17
**Scope**: Audit the *reward-design* layer of Mol-Metal against 4
pitfalls from the user's brief.
**Audit only — no code modifications.**

**Files read**:
- `molmetal/molmetal_lam/search_alg/proof_search.py` (5574 LOC) — `RewardAggregator`
  dataclass + `__call__` aggregation logic + `__post_init__` env-var overrides
  (lines 565–870, 1482–1524, 816–864 `diversity_bonus`, 3567–3591 rank-time
  diversity injection, 3559–3571 candidate filtering)
- `molmetal/molmetal_lam/search_alg/pareto.py` (440 LOC) — Pareto / NSGA-II
  dominance, hypervolume, crowding, `rank_population`
- `molmetal/molmetal_lam/sbdd_env/*` — 30 adapters, audit table from
  `molmetal/reports/wf_wire_clone_scoring.md`
- `molmetal/reports/wf_sa_penalty.md` — SA penalty channel wiring
- `molmetal/reports/wf_wire_posebusters.md` / `wf_wire_poseb_back.md` — PB channel
- `molmetal/reports/wf_pitfall_audit/p2_generator_arch.md` — upstream P2 audit
- `molmetal/molmetal_lam/tests/test_reward_aggregator.py` — coverage survey

---

## Verdict per pitfall

| ID  | Pitfall                                                                | Status      | Evidence (file:LOC) |
| --- | ---------------------------------------------------------------------- | ----------- | ------------------- |
| P3.1 | Vina-only reward → score farming, ignoring ADMET/tox/solubility/ metabolic-stability | **PARTIAL** | `proof_search.py:1482-1524` `weight_for_channel` lists 14 channels, but `proof_search.py:707-708` shows `w_vina=1.0`, `w_sa=1.0` as default-equal-weights. `proof_search.py:1526-1580` `__call__` defaults to additive aggregation. `__post_init__` (`proof_search.py:866-881`) flips `w_admet=1.0` only when env-var `ADMET_WEIGHT ≥ 1.0`; all other channels default to opt-in (`w_pb_valid=0.0`, `w_synth=0.0`, `w_anticancer_index=0.0`, `w_metal_geom_soft=0.0`, `w_platinai=0.0`, `w_rxnflow=0.0`, `w_diversity=0.0`). |
| P3.2 | Novelty / diversity / patent risk not in reward → likely fall in patent range | **PARTIAL** | `proof_search.py:777-787` `w_diversity` rank-time bonus is **opt-in** (default 0.0). `proof_search.py:816-864` `diversity_bonus` is a *binary* canonical-SMILES-uniqueness indicator (returns 1.0 if unique, 0.0 otherwise) — it is *NOT* a fingerprint-Tanimoto novelty measure against a reference corpus. `proof_search.py:3567-3591` shows the bonus is only applied at rank-time *after* MCTS finishes. No `ChEMBL` / `SureChEMBL` patent substructure search; no `MaxSim(train) ≤ τ` novelty gate; no IntDiv panel as a hard reward. |
| P3.3 | Multi-objective reward / Pareto / multi-objective RL                  | **OPEN**    | `molmetal/molmetal_lam/search_alg/pareto.py:39-440` exports `dominates`, `non_dominated_set`, `pareto_front`, `hypervolume`, `_crowding_distance`, `rank_population`. **Zero import sites**: `grep -rn "from molmetal_lam.search_alg.pareto\|from .pareto\|import pareto" molmetal/` returns 0 hits. The aggregator (`proof_search.py:1482-1524`) is **pure additive weighted-sum**; no Pareto-rank operator is consulted anywhere in the MCTS loop (`proof_search.py:2772 search`, `:4396 rollout`, `:4549 backprop`). |
| P3.4 | Reward hacking audit — detect if Lambda is gaming the reward          | **PARTIAL** | `proof_search.py:3559-3571` candidate-set uniqueness is enforced *only via the rank-time bonus* (P3.2), so MCTS-backprop itself is not gated. No `trivial-smiles` detector (e.g. emit `[Pt]` or empty molecule → +0 SA +0 Vina-proxy collapse). No `canonical_only` alarm: a candidate returning a fixed canonical SMILES every rollout would silently rank #1 forever. `proof_search.py:2657-2669` `leaf_value_var` / `PUCT_EXPLOIT_RATIO_VAR` exist but they measure the *variance* of leaf values, not *constant-identity collapse*. |

**1/4 OPEN, 3/4 PARTIAL, 0/4 AVOIDED**. The aggregator has 14
channels but 9 of them default to weight=0.0 (opt-in); the additive
weighted-sum is the only operator (no Pareto); diversity is binary +
rank-time; reward hacking is monitored only via leaf-value variance,
not via canonical-SMILES-set size.

---

## P3.1 — Vina-only reward → score farming — **PARTIAL**

### What we ship

**Inventory of all reward channels** (proof_search.py:614-706, channel
list; 707-787, weight list):

| Channel | Default weight | Source / role | Available where? |
| --- | --- | --- | --- |
| `r_vina` | **1.0** | AutoDock-Vina score (kcal/mol, negated) | built-in (CPU Vina / QVina / GPU OpenCL Vina) |
| `r_sa` | **1.0** | Ertl SA score, inverted (`1 - (s-1)/9`) | built-in (RDKit-only fallback `sa_score.py`) |
| `r_qed` | **1.0** | QED drug-likeness | built-in (`qed_scorer.py`) |
| `r_vina_proxy` | **1.0** | Cheap Vina proxy (regression model) | built-in |
| `r_posebusters` | **1.0** | PB pass-rate in [0,1] | wired via `--pb-check` (5 tests, see `wf_wire_posebusters.md`) |
| `r_pb_valid` | **0.0** (opt-in) | PB outer-gate binary | opt-in via `PB_WEIGHT` env or `w_pb_valid=1.0` |
| `r_pic50` | **1.0** | Learned pIC50 predictor (D-MPNN retrain, TODO-18) | built-in (`pic50_predictor.py`); honest negative result (pearson_r=0.181) |
| `r_retro` | **1.0** | Retrosynthesis feasibility | built-in (`retrosynthesis.py`) |
| `r_reinvent4` | **1.0** | REINVENT4 multiproperty scorer | `register_reinvent4_multiproperty_channel` (1059-1067) |
| `r_synth` | **0.0** (opt-in) | Synthesis-yield oracle | opt-in (`w_synth=1.0`) |
| `r_admet` | **0.0** (opt-in, env-var flips) | Lipinski desirability | `__post_init__` env-var `ADMET_WEIGHT≥1.0` (proof_search.py:874-881) |
| `r_anticancer_index` | **0.0** (opt-in) | TODO-15 equal-weight anticancer | opt-in (`w_anticancer_index=1.0`) |
| `r_metal_geom_soft` | **0.0** (opt-in) | soft Pt-CN prior | opt-in (`w_metal_geom_soft=1.0`) |
| `r_platinai` | **0.0** (opt-in) | PlatinAI kNN Tanimoto oracle | opt-in (`w_platinai=1.0`) |
| `r_rxnflow` | **0.0** (opt-in) | RxnFlow template match | opt-in (`w_rxnflow=1.0`) |
| `r_diversity` (rank-time only) | **0.0** (opt-in) | rank-time canonical-SMILES uniqueness | opt-in (`w_diversity=0.1`) |

### What this means

**Default reward = Vina + SA + QED + Vina-proxy + PB + pIC50 + retro
(7 channels, all weight 1.0).** ADMET, synth, anticancer-index,
metal-geometry-soft, PlatinAI, RxnFlow, diversity, PB-outer-gate
(8 channels) are *opt-in*. Concretely:

- The "headline default" in `r4_lambda_only_run.py:869-939`
  `build_lambda_only_aggregator` does *not* opt into any of them by
  default. Only the Vina-style baseline (Vina + SA + QED + PB +
  pIC50 + retro) is on.

- ADMET is the only one auto-flipped via env-var
  (`proof_search.py:874-881`); all others stay at 0.0 unless the
  caller passes `w_*` explicitly.

**Why this is only PARTIAL and not AVOIDED**: 5 of 8 supplementary
channels are *physically wired and unit-tested*, but the **default
production reward is still essentially "Vina + 4 cheap properties"**,
which is the textbook Vina-farming setup. The miti­gation is
*opt-in-by-script* rather than *opt-out-by-default*. A user running
the CLI out-of-the-box gets Vina-farming unless they know about
`--reward-admet-weight` (does not exist in `r4_lambda_only_run.py:1487-1596`
argparser), `--reward-platinai-weight` (CLI flag shipped per
platinum-task #917), `--reward-synth-weight` (does not exist), etc.

### Required-channel audit

The user's brief asks "is there a minimum required set of orthogonal
channels? Or all-optional?". **Honest answer: all-optional, no
minimum.** The aggregator is happy with `w_vina=1.0` and nothing else.
Specifically:

- ADMET (`r_admet`): **no** mandatory channel; env-var gated
- PB outer-gate (`r_pb_valid`): **no** mandatory channel; weight 0.0
- Synthesizability oracle (`r_synth`): **no** mandatory channel; weight 0.0
- pIC50 (`r_pic50`): **yes**, weight 1.0 default (but the predictor
  itself returns honest negative-result quality; pearson_r=0.181 ≤ ridge 0.572,
  see TODO-18 close-out)
- Diversity (`r_diversity`): **no** rank-time bonus unless w_diversity > 0
- Toxicity (`r_herg_proxy`): exists in
  `register_anticancer_channels` (proof_search.py:883-915), but the
  aggregator never calls `register_anticancer_channels` by default —
  the channel is *dead* in the default build.

### What we *should* do (PARTIAL → AVOIDED)

1. **Make ADMET / PB-outer-gate default-on** in
   `build_lambda_only_aggregator` (`r4_lambda_only_run.py:869-939`):
   flip `w_admet=1.0`, `w_pb_valid=1.0`, plus a sensible
   `w_herg_proxy=0.5`. The PB outer-gate is already a *gate*
   (binary), so weight 1.0 is a sensible default — non-PB-pass
   molecules earn 0.0, PB-pass earn 1.0.

2. **Add a `--reward-min-channels` CLI flag** that errors out if
   fewer than N orthogonal channels (e.g. Vina + ADMET + PB + pIC50)
   are non-zero. Default N=3.

3. **Document the "score-farming" risk** in paper §4.1: which cells
   have Vina < -7 kcal/mol but QED < 0.3 or SA > 6.5 (Vina-farming
   fingerprint). Honest framing: this is a known failure mode for
   additive weighted-sum rewards (Krenn 2020, Popova 2018).

---

## P3.2 — Novelty / diversity / patent risk not in reward — **PARTIAL**

### What we ship

1. **Rank-time diversity bonus** (`proof_search.py:777-787`,
   `:816-864`, `:3567-3591`): when `w_diversity > 0`, candidates
   that have a *unique canonical SMILES within the candidate set*
   earn `+w_diversity` (default 0.1) on top of their aggregated
   reward. This is **rank-time, not MCTS-in-loop**, so it only
   breaks ties when reward scores are within ~0.1 of each other.

3. **Aggregate diversity panels** (output metrics, not reward
   channels): `diversity_tanimoto_mean` (RDKit Morgan-ECFP4 Tanimoto),
   `diversity_homotype_mean` (homotype-distance metric, see
   `wf_lambda2_homotype_audit.md`). These are *diagnostic* outputs
   — they do not enter the reward function.

4. **No novelty channel**: there is no
   `novelty = max(0, τ − max_sim_to_training_set(c))` reward.
   `grep -rn "patent\|surechembl\|chembl_max\|max_sim"` in
   `molmetal/` returns 0 hits (verified).

5. **No scaffold-diversity reward**: the `per_residue_diversity.py`
   module exists (sbdd_env/) but is not registered into the
   aggregator's `weight_for_channel` dict
   (`proof_search.py:1494-1509`).

### Why this is PARTIAL

- The diversity bonus is **binary** (unique / not-unique within the
  result set), not a continuous Tanimoto or scaffold-diversity
  measure. If MCTS produces 5 unique molecules all with Vina=-7.0
  kcal/mol and 5 duplicate copies of one, the duplicates all get
  `+0.0` and the uniques get `+0.1`. This breaks *intra-pool*
  duplicates, but does **not** prevent the pool from being
  near-duplicates of a known drug.

- The rank-time bonus is computed **after** MCTS finishes, so MCTS
  backprop still sees the un-modified reward. Diversity is not
  *learned*, only *ranked*.

- There is no `MaxSim(train)` gate. A molecule with Tanimoto 0.95 to
  cisplatin (i.e. basically cisplatin with one extra H) is rewarded
  the same as a structurally novel compound, because the search space
  never sees a *training* or *patent* reference set.

### Patent-risk evidence (negative)

`grep -rn "patent\|surechembl"` against the entire `molmetal/`
tree returns 0 hits. We do not compare against SureChEMBL,
PubChem, or any patent corpus.

### What we *should* do (PARTIAL → AVOIDED)

1. **Add a continuous novelty channel**: `r_novelty(state) =
   max(0, τ − max_sim_to_known_drugs(state))` with τ=0.4
   Tanimoto threshold (Polykovskiy 2020 GuacaMol benchmark standard).
   Reference corpus: DrugBank + ChEMBL subset (cite-only — we don't
   need full chemical registration, just kNN Tanimoto).

2. **Wire scaffold diversity**: add the existing
   `per_residue_diversity.py` module to the
   `weight_for_channel` dict (proof_search.py:1494-1509) so
   `w_scaffold_diversity` becomes a first-class channel.

3. **Lower the default `w_diversity` weight** from opt-in
   (`0.0`) to a small default (e.g. `0.05`) so the singleton-attractor
   failure mode (cisplatin-only collapse) is auto-mitigated.

4. **Cite-only novelty panel in paper §4**: add a 1-row table
   showing how Mol-Metal's MaxSim-to-cisplatin compares to
   TargetDiff / Pocket2Mol published values (which we can pull
   from `wf_3_citeonly_sota.tex`).

---

## P3.3 — Multi-objective reward / Pareto / multi-objective RL — **OPEN**

### What we ship

`molmetal/molmetal_lam/search_alg/pareto.py:39-440` is a **fully
implemented, unit-tested NSGA-II library**:

- `dominates(a, b)` (line 39) — strict Pareto dominance, Deb 2002 §III.A
- `non_dominated_set(pop)` (line 77) — rank-0 front indices
- `pareto_front(pop)` (line 102) — front vectors
- `hypervolume(front, ref)` (line 116) — 2-D exact + D-D inclusion-exclusion
- `_crowding_distance(front_idx, pop)` (line 328) — NSGA-II crowding
- `rank_population(pop, weights=...)` (line 367) — full (Pareto rank, crowding, weighted scalar) sort

This is a **production-quality 440-line library** with proper Deb
2002 / Zitzler 1999 / Knowles 2006 literature citations in the
module docstring.

### Critical observation: zero call sites

```bash
$ grep -rn "from molmetal_lam.search_alg.pareto\|from .pareto\|import pareto" molmetal/
# 0 results
```

`pareto.py` is **never imported anywhere in the codebase**. The
aggregator (`proof_search.py:1482-1524`) is a *pure additive
weighted-sum*; the rank-time diversity bonus is a flat +0.1 bonus,
not a Pareto-rank operator; the MCTS loop (`:2772`, `:4396`,
`:4549`) does not consult Pareto rank at any point.

### Why this is OPEN

This is a textbook multi-objective failure mode. The Mol-Metal
reward has at least 7 default-active scalar channels (Vina + SA +
QED + PB + pIC50 + retro + the rank-time diversity + the optional
ADMET + pIC50). The Pareto front of these 7 objectives is
inherently non-convex and the additive weighted-sum collapses to a
single point — *no* Pareto exploration happens.

Concrete consequences:
- A molecule with `Vina=-7.0, SA=4.5, QED=0.3, PB=1.0, pIC50=8.0`
  earns `1·(7.0)+1·(1-3.5/9)+1·0.3+1·1.0+1·8.0 = 16.9` and
  dominates a molecule with `Vina=-7.5, SA=8.0, QED=0.1, PB=0.0,
  pIC50=4.0` which earns `1·7.5+1·(1-7/9)+1·0.1+0+1·4.0 = 11.7`.
  Additive ranking picks the Vina-farming high-Vina-but-bad-drug
  compound.

- The Pareto-optimal point for {Vina, QED, SA} might be
  `{Vina=-6.5, QED=0.6, SA=3.5}` (decent drug with reasonable
  binder); the additive sum picks `{Vina=-7.5, QED=0.3, SA=8.0}`
  (highly lipophilic, hard to make, bad druglikeness). This is
  the canonical motivation for NSGA-II (Deb 2002).

### What we *should* do (OPEN → AVOIDED)

1. **Wire `rank_population` into the candidate-ranker** at
   `proof_search.py:3559-3593`: replace the additive
   `candidates.sort(key=lambda p: p[0], reverse=True)` with
   `candidates = rank_population(population, weights=...)`
   over the 5-7 active channels. Cost: O(k²·d) for ≤ k=20
   top-K candidates, negligible (<1 ms).

2. **Add a `--postprocess-pareto` CLI flag** to
   `r4_lambda_only_run.py` argparser that toggles Pareto-rank
   post-sort vs additive sort. Default OFF (backward compat);
   Round-14+ turn it ON.

3. **Cite pareto.py in paper §4**: add 1 paragraph in §4.7
   (multiobjective ablation) showing the Pareto front of the
   5-channel reward for the 100-pocket × 3-seed sweep. Lit
   anchor: Deb 2002 NSGA-II, Knowles 2006 hypervolume.

4. **Add a Pareto ablation row** to paper §5 ablation:
   `--postprocess-pareto on vs off` on a 10-pocket × 3-seed
   subset, measuring diversity of the top-K result set (IntDiv,
   scaffold diversity). Honest projection: +5-15pp IntDiv at
   ≤ −0.1 mean reward cost (Deb 2002 non-dominated set
   typically dominates weighted-sum on diversity metrics).

---

## P3.4 — Reward hacking audit — **PARTIAL**

### What we ship

**Lambda is MCTS, not RL** (P2.1 already documented in
`wf_pitfall_audit/p2_generator_arch.md` lines 42-83). MCTS does
not have a policy that can be *trained* to game the reward. The
risk surface is *not* "agent learns to exploit reward bugs" but
**"MCTS converges on a single high-reward canonical SMILES"** (the
exact failure mode we already saw in
`WF-Lambda-Fix-Singleton`: 30/30 cells collapsed to one candidate).

Three mechanisms ship:

1. **Rank-time diversity bonus** (proof_search.py:816-864, 3567-3591) —
   the singleton-attractor breaker. Default weight 0.0 (opt-in).

2. **`tree_diversity` panel** (proof_search.py:3631-3633): the
   *ratio* of unique canonical-SMILES leaves over explored nodes.
   Recorded per-iteration; used as a diagnostic. **Not used as a
   reward signal** — only as a report column.

3. **`leaf_value_var` / `leaf_value_std`** (proof_search.py:3638-3645):
   the variance of leaf values across the search. If all leaves
   converge to the same reward (because all canonical SMILES
   collapse), this *should* trend to 0.0. **But** if MCTS collapses
   to one SMILES that earns a single fixed reward, variance is
   literally 0 and the alarm doesn't fire (because the *correct*
   alarm is "unique-canonical-SMILES-count = 1", not "variance = 0").

### What is missing (PARTIAL → AVOIDED)

1. **No `trivial_smiles` detector**: if the MCTS returns only
   `[Pt+2]`, `[Pt]([Cl])([Cl])`, `O`, or empty molecules, no
   alarm fires. The user gets a JSON saying `n_distinct=1` but
   nothing in the aggregator recognises this as "reward hacking
   on the seed".

2. **No `canonical_only` alarm**: if every iteration produces
   *exactly the same canonical SMILES*, the rank-time bonus
   returns 0.0 to every candidate (because none is unique), but the
   *reward* still rewards the seed. The MCTS does not recognise
   this as a failure.

3. **No `max_sim_to_seed` gate**: a candidate with Tanimoto 1.0
   to the metal-seed (i.e. literally the seed with one H added)
   earns the same reward as a structurally novel compound. The
   Singleton-Attractor failure mode (WF-Lambda-Fix-Singleton,
   Round-12) was *exactly* this: 30/30 cells emitted
   `[NH2][Pt]([NH2])([Cl])[Cl]` (cisplatin) regardless of seed.

4. **No `n_distinct` audit**: even though
   `r4_lambda_only_run.py` records `n_distinct` as a metric, the
   aggregator does not *react* to it. A cell with `n_distinct=1`
   should trigger a low-confidence flag.

### What we *should* do (PARTIAL → AVOIDED)

1. **Add a `r_anti_collapse` channel**: `r_anti_collapse(state) =
   1.0` if the candidate's canonical SMILES has Tanimoto ≤ 0.7 to
   the metal-seed AND ≤ 0.7 to every other member of the
   candidate-set, else 0.0. Wire via `w_anti_collapse=0.5` default
   ON in `build_lambda_only_aggregator`.

2. **Promote `tree_diversity` to a hard alarm** in the closed-loop
   pipeline: if `tree_diversity < 0.05` (i.e. <5% unique
   canonical-SMILES leaves), emit a `MCTS_COLLAPSED` warning and
   abort the run. Logged at WARN, not ERROR (so the user can still
   see the result), but flagged in the summary JSON.

3. **Add a `n_distinct` minimum gate**: if `n_distinct < 3`,
   record `low_diversity_warning=true` in the JSON summary and
   exclude the cell from aggregate metrics that require
   diversity (e.g. IntDiv panel). Cheap; just a post-filter.

4. **Document the failure mode in paper §6 limitations**: add a
   paragraph describing the singleton-attractor failure mode
   (3-layer collapse: chemistry / MCTS cache / reward prior),
   cite WF-Lambda-Fix-Singleton and WF-Lambda-Fix-FullPath-v2.

---

## Cross-cutting observations

- **24+ reward channels available, 8 actively-on by default.** The
  aggregator has been *exhaustively instrumented* (proof_search.py
  ships 14 channel / 14 weight dataclass fields); what is missing
  is the **default-policy**: which channels should be ON without
  the user having to know they exist. Recommend flipping
  ADMET / PB-outer-gate / hERG-proxy / diversity to default-on in
  `build_lambda_only_aggregator` (r4_lambda_only_run.py:869-939).

- **No multi-objective operator.** Despite a 440-line NSGA-II
  library sitting in `pareto.py` unused, the MCTS loop runs
  additive weighted-sum. Pareto-rank post-processing is a
  low-effort, high-leverage lift (paper §4.7 ablation row + 1-line
  CLI flag).

- **No novelty/patent gate.** The closest mitigation is the
  rank-time diversity bonus, which only fires when canonical
  SMILES *within the result set* collide. There is no
  Tanimoto-to-known-drugs gate, no SureChEMBL patent search, no
  MaxSim(train) channel. This is a meaningful gap for any
  production use.

- **No active reward-hacking detector.** The `tree_diversity`
  diagnostic fires *after* the fact; no channel actively penalises
  singleton collapse. Recommend adding `r_anti_collapse` and
  promoting `n_distinct` to a gate.

---

## Honest framing

What we *did* ship is real and useful:

- 14 reward channels physically wired (5 default-on, 8 opt-in, 1 rank-time).
- SA penalty channel verified (mean -0.043 on rich pools, QED tradeoff well under 0.05).
- PB channel wired with 5 tests (`wf_wire_posebusters.md`), outer-gate variant
  available via `w_pb_valid`.
- ADMET env-var override (`__post_init__` flips `w_admet=1.0`).
- Rank-time diversity bonus breaks intra-pool duplicates.
- `tree_diversity` and `leaf_value_var` diagnostics exposed per-iteration.

What we *did not* ship:

- A default policy that turns on ADMET / PB-outer-gate / hERG /
  diversity without the user knowing.
- A novelty channel (MaxSim-to-known-drugs).
- A Pareto-rank operator in the candidate-sorter (despite a
  440-line NSGA-II library sitting in `pareto.py` unused).
- An active reward-hacking detector (`r_anti_collapse`,
  `n_distinct` gate, `MCTS_COLLAPSED` alarm).

Net verdict: the *plumbing* is in place; the *policy* is not. The
audit ships all four pitfalls with concrete patch plans; the
implementations are intentionally out of scope per "audit only —
no code modifications" constraint.

---

## Files referenced (absolute paths)

- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/pareto.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/sa_score.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/cite_only_sota_comparator.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/pic50_predictor.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/retrosynthesis.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_sa_penalty.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_wire_clone_scoring.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_wire_posebusters.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_fix_singleton/final.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pitfall_audit/p2_generator_arch.md