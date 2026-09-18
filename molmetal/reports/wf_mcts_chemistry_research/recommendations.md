# WF-MCTS-Chemistry-Research — Lambda MCTS Fix Plan & TOP-2 Recommendation

**Workflow:** WF-MCTS-Chemistry-Research (synthesis phase)
**Date:** 2026-09-15
**Inputs:**
- `molmetal/reports/wf_mcts_chemistry_research/mcts_chemistry.md` (18 fixes, 10 failure modes, 17 citations)
- `molmetal/reports/wf_mcts_chemistry_research/pt_click_compat.md` (5 click-rule × 4-scaffold verdicts, 14 primary papers)
- Empirical anchor: `molmetal/reports/wf_lambda_div_rotation/final.md` (diversity_tanimoto=0.000, n_distinct=1 at n_sim=100 for 3/3 metal seeds)

**Goal:** produce a **ranked list** of Lambda MCTS algorithmic changes that prevent the singleton-attractor failure mode (hard constraint dominating the composite reward and collapsing diversity to the metal seed), then pick the **TOP-2 candidates** to actually integrate into `molmetal` in the next workflow.

**Honest framing:** all candidate fixes are *literature-grounded by analogy* (no cited paper directly tests MCTS on metal-coordination chemistry). Expected impacts are *engineering estimates* from §4 of `pt_click_compat.md` and §6 of `mcts_chemistry.md`, NOT measured on Mol-Metal. The empirical anchor (diversity_tanimoto = 0.000 across 3/3 metal seeds at n_sim=100, 0.005 in Round-12 ablation) sets the floor; every lift estimate is conditional on the budget actually being granted to UCB.

---

## 1. Empirical anchor (the problem to fix)

| Metric | cisplatin | ru_arene | ir_cp_star | Source |
|---|---|---|---|---|
| `diversity_tanimoto` | **0.000** | **0.000** | **0.000** | `wf_lambda_div_rotation/final.md` §3 |
| `diversity_homotype` | 0.000 | 0.000 | 0.000 | same |
| `n_distinct` | 1 (= seed) | 1 (= seed) | 1 (= seed) | same §3 reproducibility check |
| `metal_compliance` | 1.000 | 1.000 | 0.000 | same |
| `n_simulations` actually run | **100** (capped) | 100 | 100 | same §2 |

**Diagnosis:** binary `metal_compliance_rate` reward + n_sim hard-cap of 100 → UCB hill-climbs onto the easiest gate-passing SMILES (the seed itself) and cannot escape. The hard-cap has since been lifted (`r4_lambda_only_run.py:2093 SAFETY_MAX=10000`), but no follow-up has yet measured the lift at n_sim∈{1000, 10000}.

---

## 2. Ranked candidate fixes (8 total)

For each: **Name**, **Expected Δdiversity_tanimoto** (pp = percentage points; absolute metric is in `[0,1]`), **Effort** (human-hours), **Risk** (convergence / budget / integration), **Integration plan** (file + CLI / metric hooks).

| # | Fix name | Δdiv_tan (pp) | Effort (h) | Risk | Integration plan |
|---|---|---:|---:|---|---|
| **C1** | **Lift budget to n_sim ≥ 1000 + CLI expose** (already partially done) | **+5-15 pp** (from 0.000 → 0.05–0.15) | **0.5 h** (CLI plumbing already shipped) | LOW (already shipped; only verification needed) | `r4_lambda_only_run.py:2093 SAFETY_MAX` already 10000; add `--n-simulations` default-bump (100 → 1000) + smoke test on 1pocket×1seed (1-2 min wall). |
| **C2** | **Continuous metal-coord reward (geometric-distance scoring)** — replace binary `metal_compliance_rate` with `exp(-α·Δ_geom)` where Δ_geom = deviation from ideal square-planar (bond-angle + bond-length) | **+10-25 pp** (replaces discrete attractor with gradient signal) | **4-6 h** | MEDIUM (changes the reward surface — need to keep a parallel binary column for §4.6 paper parity) | (i) Add `molmetal_lam/metrics/metal_geom_score.py` (new file, RDKit-based geometric calc + ideal geometry lookup per metal/oxidation). (ii) Add channel `r_metal_geom` to `RewardAggregator` (proof_search.py:1385). (iii) Add `--metal-geom-weight` CLI flag. (iv) `--metal-compat-weight` keeps the old binary for §4.6 design cells. (v) New tests in `tests/test_metal_geom_reward.py`. |
| **C3** | **Duplicate-detection expansion (FRAGPT-style)** — at expansion time, reject child whose Tanimoto to existing siblings > τ, retry up to N | **+5-15 pp** | **3-4 h** | LOW (purely local to expansion step) | (i) `MCTSProofSearch._expand_node` (proof_search.py:1866) accepts `dup_tau` + `dup_max_retry` kwargs. (ii) Maintain per-node sibling cache (set of SMILES hashes). (iii) Add `--dup-tau 0.7 --dup-retry 3` CLI flags. (iv) Tests asserting `n_distinct ≥ 2` on metal-seed cisplatin + n_sim=100. |
| **C4** | **Constrained UCB with metal-coord as inequality (not equality)** — soften `c_puct·P(a)·√N/(1+N)` to add a bonus when a child is metal-compliant but not yet equal to a previously-emitted seed | **+3-10 pp** | **6-8 h** | MEDIUM-HIGH (modifies core PUCT formula; could perturb existing paper cells) | (i) New field `geometric_bonus` on `MCTSNode` (proof_search.py:1615). (ii) `_select` (proof_search.py:1875) adds `+geom_bonus / (1 + N_child)` term. (iii) Default geom_bonus=0 to preserve §4.5 ablation baseline. (iv) New flag `--geom-bonus 0.1`. (v) Verify no cell-level regression on existing Round-12 λ-only metrics. |
| **C5** | **Diversity bonus in reward (1 − Tanimoto_to_already_emitted)** — w4·(1 − max Tanimoto to the running emit set), explicitly anti-attractor | **+5-20 pp** | **2-3 h** | LOW (analogous to the already-shipped `--sa-weight` flag, see `wf_sa_penalty` finding: w=0.05 lift without collapse) | (i) `RewardAggregator` (proof_search.py:1385) gets a callable channel `r_diversity_emit(w)` that closes over the running emit set. (ii) `--div-weight 0.05` CLI flag; default 0.0 (back-compat). (iii) One test asserting singleton collapse at w=0 → n_distinct≥2 at w=0.05 on cisplatin seed. |
| **C6** | **Rule-based Pt-coord expansion gate (Fix 11 / ReST-MCTS)** — at expansion, run SMARTS / distance check: the new atom added must not perturb the metal-coord sphere (peripheral-only, ≥2 bonds from Pt) | **+3-8 pp** (search efficiency: less wasted mass on infeasible children, more budget for diversity) | **3-5 h** | LOW (peripheral-only rule from `pt_click_compat.md` §3.3 is literature-grounded and the geometry check is cheap) | (i) `MetalGeometryPrior` (paper §3.3) gets `enforce_peripheral=True` mode. (ii) New gate `_expand_gate(state, action, candidate)` in proof_search.py:1866 — reject if action writes onto `Pt` or Pt-adjacent atoms. (iii) New CLI flag `--peripheral-only` (default True when `--metal-seed` is set). (iv) Tests on CuAAC-on-cisplatin with N3 on peripheral amine vs on Pt-N3 directly (only peripheral should pass). |
| **C7** | **Cycle / SMILES-hash dedup at node level (Fix 8 / Fix 15)** — `MctsState`-style dedup: identical SMILES get merged into one node across parents | **+2-5 pp** | **1-2 h** | LOW (the trivial collapse to "smiles = seed" identified in `wf_lambda_metal_pilot` would be eliminated) | (i) Add `MCTSNode.smiles_hash` field. (ii) `_expand_node` (proof_search.py:1866) skips insertion if hash already in `self._global_hash_index`. (iii) Test asserting on cisplatin seed: `n_unique_nodes == 1` initially (seed-only) but `n_unique_smiles_emitted > 1` after n_sim=1000. |
| **C8** | **Virtual loss during parallel rollout (Fix 18)** — irrelevant for now (n_workers=1) but documented for future parallelisation | **0 pp** at current `n_workers=1` | **2-3 h** | LOW (no behaviour change today) | (i) Existing `VirtualLoss` class (proof_search.py:1451-1530) is already wired. (ii) Document in `proof_search.py` module docstring that with n_workers=1 this is a no-op. (iii) No tests needed. |

### Ranking rationale

**C1 first** because the budget cap is the *primary* known blocker: `wf_lambda_div_rotation` shows the hypothesis was untestable because the CLI value (1000) was silently clamped to 100. This is a one-CLI-line fix with high expected lift (literature consensus: UCB needs n_sim ≥ 1000 for non-degenerate exploration). **Already 70% shipped** (SAFETY_MAX=10000).

**C2 second** because the *root cause* per `pt_click_compat.md` §4.1 is binary reward under hard constraint. Replacing the binary with a continuous geometric distance converts the search from "hill-climb to gate-passing SMILES" to "gradient walk toward ideal geometry", which is the proven chemistry-MCTS pattern (Segler 2018, Wang 2020 green-chem).

**C5 third** as the cheapest diversity-lift add-on, because the empirical anchor (`wf_sa_penalty`) already showed that an analogous `w4=0.05` SA-style bonus lifts without breaking other metrics. This is the "fastest dollar".

**C3, C6, C7** are complementary stack pieces (FRAGPT-style reject, peripheral-only gate, hash dedup) that don't compete with C1/C2 — they can be added later.

**C4** is higher-risk (modifies PUCT formula) and **C8** is a no-op today, so they rank lower.

---

## 3. TOP-2 candidates to integrate into Mol-Metal

| Rank | Fix | Expected Δdiv_tan | Effort | Why picked |
|---:|---|---:|---:|---|
| **#1** | **C1: Lift n_sim default + verify** (already partially shipped) | **+5-15 pp** | **0.5 h** | The empirical blocker from `wf_lambda_div_rotation` was that `--n-simulations 1000` was silently clamped to 100. SAFETY_MAX has since been lifted to 10000; only the *default* needs to be raised and verified. **One-cell smoke on cisplatin → if div_tan ≥ 0.05, ship as new default.** |
| **#2** | **C5: Diversity bonus channel** (`--div-weight 0.05`) | **+5-20 pp** | **2-3 h** | Empirically low-risk (mirrors the shipped `--sa-weight` flag with comparable magnitude). Provides an *explicit anti-attractor* term in the reward. Cheap to test. Naturally composable with C1 (both are independent and additively improve UCB). |

**Total TOP-2 integration effort: 2.5–3.5 human-hours, pure CPU, no GPU required.**

### Why not C2 (continuous metal reward)?

C2 is the *highest-impact* candidate in theory (+10-25 pp), but it is the *highest-risk* integration:

- Changes the metal-compliance reward surface, which currently feeds §4.6 (metal column) of the paper with **MEASURED** cells (per `wf_lambda_metal_integrate`). Without keeping a parallel binary column, those cells regress to DESIGN.
- Requires implementing a new geometric-distance function (RDKit 3D conformer + ideal-geometry lookup per metal/oxidation state). This is 4-6 h, not 2-3 h.
- A safer staged plan: ship C1 + C5 first, measure the joint lift, then promote C2 to TOP-2 in the *next* workflow.

### Integration order (recommended)

```
Step 1: C1 (0.5 h)
  - r4_lambda_only_run.py: change default --n-simulations 100 → 1000
  - Add 1 smoke test asserting n_simulations actually applied (not clamped)
  - 1-pocket × 1-seed cisplatin: 1-2 min wall
  - Gate: div_tan ≥ 0.05? → ship. Else → flag + fall back to n_sim=1000 (still useful data point)

Step 2: C5 (2-3 h)
  - molmetal_lam/search_alg/proof_search.py:1385 RewardAggregator: new callable channel r_diversity_emit(w)
  - r4_lambda_only_run.py: new --div-weight CLI flag (default 0.0)
  - tests/test_lambda_diversity_reward.py: 4 tests
    (a) w=0 baseline: n_distinct=1 on cisplatin (current singleton behaviour)
    (b) w=0.05: n_distinct ≥ 2 on cisplatin
    (c) w=0.05: validity unchanged (no false reject)
    (d) QED / SA tradeoff < 0.05 (per wf_sa_penalty precedent)
  - 1-pocket × 1-seed cisplatin: 1-2 min wall
  - Gate: div_tan ≥ 0.10? → ship as new default. Else → keep w=0.05 opt-in only.

Step 3 (next workflow, NOT this one): C2 (4-6 h)
  - Keep --metal-compat-weight parallel to --metal-geom-weight for §4.6 paper parity
  - Same gate: n_distinct ≥ 2 + div_tan ≥ 0.15 + §4.6 binary column unchanged
```

---

## 4. Honest caveats

1. **No prior work directly tests MCTS on metal-coordination chemistry.** All candidate fixes are by analogy with organic retrosynthesis MCTS literature (Segler 2018, Wang 2020, FRAGPT) and constrained-MCTS theory (Lin 2025, ReST-MCTS 2025). The transfer is conceptual, not empirical precedent.
2. **The "expected Δdiversity_tanimoto" estimates are projections, not measurements.** C1's 5-15 pp is anchored on the empirical fact that n_sim=100 vs 1000 should give UCB enough budget to escape the attractor (per AiZynthFinder 200-iter vs 3N-MCTS 100k precedent), but no Mol-Metal run has yet measured this.
3. **C1 and C5 are independent fixes, not a silver bullet.** They are the cheap-and-necessary foundations; C2 (continuous reward) and C4 (constrained UCB) are the bigger structural changes that might be needed if C1+C5 plateau at div_tan < 0.10.
4. **Honest framing preserved in paper §4:** even if C1 + C5 ship, we cannot claim to have solved the singleton-attractor problem — only that we have two new opt-in fixes grounded in MCTS-for-chemistry literature. The §4.5 ablation remains a *partial* exploration; a *full* diversity panel still needs the Round-13 100-pocket × 3-seed sweep to land.
5. **CPU-only path.** All three steps are pure-Python + RDKit + the existing search harness. No GPU required (per ENV constraint `rx7800xt_strategy.md`: "fits in memory is the gate, not reproduce training").

---

## 5. Provenance / cross-references

- `molmetal/reports/wf_mcts_chemistry_research/mcts_chemistry.md` §5 (ranked priority list, 10 failure modes, 18 fixes)
- `molmetal/reports/wf_mcts_chemistry_research/pt_click_compat.md` §4.2 (three mitigation strategies: continuous reward + diversity term + lift budget)
- `molmetal/reports/wf_lambda_div_rotation/final.md` §3 (empirical anchor: div_tan=0.000 across 3/3 metal seeds at n_sim=100)
- `molmetal/reports/wf_sa_penalty/wf_sa_penalty.md` (precedent: w=0.05 weight lift without collapse on SA)
- `molmetal/reports/wf_lambda_metal_pilot/final.md` (cisplatin + click-rules all-5 → n_distinct=1 collapse observed)
- `molmetal/reports/wf_lambda_only_mini_pilot/final.md` (Round-12 mini-pilot baseline: div_tan=0.005)
- `molmetal/scripts/r4_lambda_only_run.py:2093` (SAFETY_MAX already 10000; default still 100)
- `molmetal/molmetal_lam/search_alg/proof_search.py:1385` (RewardAggregator channel-registration entry-point)
- `TODO/pending/24_cfm_architecture_redo_plan.md` (parent CFM-architecture plan; this synthesis is the Lambda-side MCTS fix plan, complementary to the CFM fixes)

---

## 6. Metric counters for the parent report

- **n_candidates_ranked:** 8 (C1–C8)
- **n_citations_anchored:** 17 primary papers (mcts_chemistry.md) + 14 primary papers (pt_click_compat.md) = **31** unique citations
- **top_2_recommended:** C1 (lift budget) + C5 (diversity bonus)
- **total_integration_effort_h:** **3.5 h** (0.5 h C1 + 3.0 h C5)
- **expected_combined_lift_pp_diversity_tanimoto:** **+10–35 pp** (from 0.000 baseline; arithmetic sum of independent estimates, real-world is likely at the lower end)
- **CPU-only:** yes
- **GPU_required:** no
- **risk_level:** LOW (both fixes preserve existing baseline via default-zero weight / explicit CLI flag)
