# Phase 2 — Prioritized Algorithmic-Tuning Task List

**Date**: 2026-09-15
**Phase**: Phase 2 (aggregate Phase 1A lit + Phase 1B code review)
**Author**: WF-Algo-Tune
**Inputs**:
- `molmetal/reports/wf_algo_tune/phase1a_lit_research.md` (lit grounding)
- `molmetal/reports/wf_algo_tune/phase1b_code_review.md` (proof_search.py + closure.py + kinetic_aggregator.py + metal_geometry.py review)

---

## 0. Scoring rubric

Each candidate scored on three axes from the directive:

| Axis | HIGH | MEDIUM | LOW |
|---|---|---|---|
| **Impact** | Breaks singleton attractor / lifts diversity / lifts PB pass | Modular improvement | Polish |
| **Risk** | May break existing behaviour | Additive, easy to revert | Isolated module |
| **Effort** | hours wall-clock | — | — |

Plus:
- **lit-anchor** — which paper / theorem
- **math-prior** — algebraic / optimisation-formulation (no hand-waving)
- **file-set** — which files modified; must be disjoint across the 5 picks

---

## 1. Candidate pool (n=7)

| ID | Name | Lit anchor | Impact | Risk | Effort (h) |
|---|---|---|---|---|---|
| C1-DPW | F3-DPW action widening (Auger 2013 §3.2) | Auger 2013 | HIGH | MEDIUM | 2.0 |
| C2-Murcko | F4-Bemis-Murcko scaffold-aware diversity | Bemis 1996 + Schuffenhauer 2007 + Polykovskiy 2020 | HIGH | LOW | 1.5 |
| C3-ClosureConsult | F5.f closure-test consult on cache-hit | Auger 2013 Thm 1 + paper/appendices/closure_theorem.tex | HIGH | MEDIUM | 2.5 |
| C4-TauEntropy | F-entropy-τ replace hard metal prior | Neu 2017 + Williams 1992 | HIGH | MEDIUM | 1.5 |
| C5-KineticDeprecate | F5.e deprecate kinetic_aggregator (PEP 387) | PEP 387 + Williams 1992 | LOW | LOW | 0.5 |
| C6-PUCT-Selective | F1 doc-only UCB→PUCT clarification | Rosin 2011 + Silver 2016/2017 | LOW | LOW | 0.3 |
| C7-ROOT-Diversity | F-scaffold-PUCT bias in _prior | Bemis 1996 + Schuffenhauer 2007 + Polykovskiy 2020 | HIGH | HIGH | 3.5 |

---

## 2. Prioritized TOP 5 picks

Selected by: high-impact/low-risk ratio, disjoint file sets, lit-grounded, math-prior, no overlap with `r4_lambda_only_run.py` (Phase 4 integrator) or the w8579x29t Phase 3 file set.

### T1 — F3-DPW Double Progressive Widening in MCTS expansion (priority 1)

- **File set**: `molmetal/molmetal_lam/search_alg/proof_search.py`
- **Lines**: edit inner loop at `_expand:2903-2919` to add DPW filter
- **Lit anchor**: Auger 2013 §3.2 — `|A(s,t)| ≤ floor(n(s,t)^α)`, α ∈ (0,1)
- **Math prior**: α=0.5 default; controls exploration/exploitation ratio
- **Impact**: HIGH (lazy branch growth attacks 5 rules × 220 tiles explosion; the singleton-attractor mode is partly caused by always-expanding all 1100 pairs per `_expand` call)
- **Risk**: MEDIUM (additive branch filter; defaults preserve legacy behaviour)
- **Effort**: 2.0 h
- **Expected lift**: +5-15pp diversity (Auger 2013 convergence-rate argument; **NOT measured** on this host — honest framing)
- **Verification**:
  - 3 tests in `test_proof_search_prior.py`: α=0.5 caps branches at `floor(n^0.5)`, α=0.0 = legacy, α=1.0 = unrestricted
  - All existing `test_proof_search_*.py` tests stay green
  - 5×1 smoke, wall-clock < 10% increase

### T2 — F4-Bemis-Murcko scaffold-aware diversity metric (NEW module) (priority 2)

- **File set** (NEW): `molmetal/molmetal_lam/metrics/scaffold_diversity.py`, `molmetal/molmetal_lam/tests/test_scaffold_diversity.py`
- **Lit anchor**: Bemis & Murcko 1996 (J. Med. Chem. 39:2887-2893) — R(G), L(G), S(G); Schuffenhauer 2007 scaffold tree; Polykovskiy 2020 IntDiv set-Jaccard
- **Math prior**: `J(S_a, S_b) = |S_a ∩ S_b| / |S_a ∪ S_b|` over scaffold SMILES; `scaffold(S) = MurckoScaffold.GetScaffoldForMol(mol)`
- **Impact**: HIGH (new diversity axis absent from Tanimoto / Homotype)
- **Risk**: LOW (new module, no call-site touched; mirrors `homotype_diversity.py` pattern)
- **Effort**: 1.5 h
- **Expected lift**: 0pp directly; enables downstream PUCT-scaffold-bias (deferred to Phase 3 — T7 risk=HIGH)
- **Verification**:
  - 6 tests in `test_scaffold_diversity.py`: identical→1.0, disjoint→0.0, isomers→non-1.0, benzene vs naphthalene→0.0 (orthogonal rings), cisplatin vs cisplatin→1.0 (identity), pytest.approx on ChEMBL set

### T3 — F5.f closure-theorem consultation on `_expand` cache-hit (priority 3)

- **File set**: `molmetal/molmetal_lam/lam_chem/closure.py`, `molmetal/molmetal_lam/search_alg/proof_search.py`
- **Lines**: insert closure-consult branch at `_expand:2898-2902` (pre-loop, non-overlapping with T1's inner-loop edit)
- **Lit anchor**: Auger 2013 Thm 1 (random-tree MCTS convergence); `paper/appendices/closure_theorem.tex` (ProductiveSpace closure_test with O(B·N·|R|) bound)
- **Math prior**: If `closure_test(seed, depth=2*max_depth)` returns 0 reachable nodes, seed is closure-dead → short-circuit expansion
- **Impact**: HIGH (kills the bare-metal seed expansion-time trap; Phase 1B §6 confirmed the constructive-synthesis guarantee from `WF-Lambda-4` is unused at runtime)
- **Risk**: MEDIUM (touches hot path `_expand`; must guard with depth-budget check)
- **Effort**: 2.5 h
- **Expected lift**: +20-40% wall-clock saving on closure-dead seeds (projection; not measured)
- **Verification**:
  - All existing closure tests stay green (6 unit + 4 property in `test_closure_theorem.py` + `test_closure_theorem_property.py`)
  - 3 tests in `test_proof_search_prior.py`: closure-dead seed → fewer candidates; closure-live seed unchanged; cache-hit logs WARN
  - 5×1 cisplatin with closure-consult=ON vs OFF, wall-clock delta measured

### T4 — F-entropy-τ replace hard metal prior with τ-temperature (priority 4)

- **File set**: `molmetal/molmetal_lam/priors/metal_geometry.py`
- **Lit anchor**: Neu 2017 (entropy-regularized MDPs) — `π* = argmax E[Σγ^t r] + τ·H(π)`; Williams 1992 REINFORCE eq. (4) for variance reduction
- **Math prior**: `exp(-τ·d(geometry_violated))`; τ→0 recovers hard gate, τ=0.5 default; continuous knob between exploitation and exploration
- **Impact**: HIGH (replaces hard 0/1 gate with continuous knob; reduces gradient variance per Williams 1992; directly attacks the middle layer of the 3-layer singleton attractor)
- **Risk**: MEDIUM (single-module edit; behind CLI flag `--metal-temp`)
- **Effort**: 1.5 h
- **Expected lift**: +3-8pp metal_compliance_rate on non-seed molecules (Williams 1992 variance-reduction bound; **NOT measured**)
- **Verification**:
  - All existing `metal_geometry.py` tests stay green
  - 4 tests: τ=0 → hard gate; τ=∞ → 0.5 uniform; τ=0.5 default returns tiered `[1.0, 0.61, 0.37, 0.13]` for distance `[0,1,2,3]`; monotonic decreasing
  - 5×1 cisplatin with `--metal-temp 0.5` vs default, metal_compliance_rate delta measured

### T5 — F5.e deprecate kinetic_aggregator with PEP-387 warning (priority 5)

- **File set**: `molmetal/molmetal_lam/reactions/kinetic_aggregator.py`
- **Lit anchor**: PEP 387 (Python deprecation policy); Williams 1992 (motivation for variance reduction)
- **Math prior**: No math change. Adds `DeprecationWarning` to module + each public function. Honest framing: legacy stub preserved for backward compat with `WF-Lambda-1` first iteration.
- **Impact**: LOW (no behaviour change in production paths)
- **Risk**: LOW (`warnings.warn` only; existing callers unchanged)
- **Effort**: 0.5 h
- **Expected lift**: 0pp by design (hygiene only)
- **Verification**:
  - All 7 existing tests in `test_kinetic_aggregator.py` stay green (decorated with `filterwarnings('ignore', category=DeprecationWarning)`)
  - 1 test: importing module triggers `DeprecationWarning`

---

## 3. Conflict check

| Task | File(s) | Conflict with |
|---|---|---|
| T1 | `proof_search.py` | T3 (different lines: T1 = inner loop 2903-2919, T3 = pre-loop 2898-2902 — **non-overlapping** verified) |
| T2 | NEW `scaffold_diversity.py`, NEW `test_scaffold_diversity.py` | none |
| T3 | `closure.py`, `proof_search.py` | T1 (different lines in `proof_search.py`) |
| T4 | `metal_geometry.py` | none |
| T5 | `kinetic_aggregator.py` | none |

**`files_shared`: []** — `none: true`

---

## 4. Expected wall-clock

| Task | Effort (h) | Cumulative (h) |
|---|---|---|
| T2 | 1.5 | 1.5 |
| T5 | 0.5 | 2.0 |
| T1 | 2.0 | 4.0 |
| T4 | 1.5 | 5.5 |
| T3 | 2.5 | 8.0 |
| **Sum** | **8.0** | — |
| + 30% buffer (sequential + test integration) | **10.4** | — |

Execution order rationale:
- T2 first: pure-additive new module, builds foundation for downstream PUCT-scaffold-bias (Phase 3 T7)
- T5 second: quickest, pure hygiene, builds momentum
- T1 third: highest expected lift, modifies inner loop, kept after T2/T5 so unrelated fixes don't conflict
- T4 fourth: replaces hard metal prior; orthogonal to T1
- T3 last: depends on T1's expanded branch set being meaningful; closure-consult makes more sense after the search tree is wider

---

## 5. Out-of-scope (do NOT execute in Phase 2)

| ID | Reason |
|---|---|
| C6-PUCT-Selective | Docstring-only fix, 0pp lift, deprioritised to documentation sweep |
| C7-ROOT-Diversity-Axis | Risk=HIGH; depends on T2's scaffold module; deferred to Phase 3 after T2 is verified |

---

## 6. Honest framing

1. **All `expected_lift` values are projections from published theorems** (Auger 2013 Thm 1, Williams 1992 eq. (12), Bemis 1996, Neu 2017 Thm 3, Polykovskiy 2020 IntDiv) — **NOT measurements on this host**. The Round-13 100×3 honest negative is the reference frame, not the lift target.

2. **T1 and T3 are MEDIUM risk**, not LOW risk. Both touch the MCTS hot path `_expand`. T3 in particular edits a line that runs every `n_simulations=1000` iterations × 5 rules × 220 tiles. Per phase1b §3 the `_expand` cost is the dominant factor.

3. **T5 has 0pp expected lift by design** — it is hygiene, not a tuning improvement. Included only because the directive requires 5 tasks and the file is genuinely under deprecation flag.

4. **No task guarantees Round-13 reversal.** The Round-13 100×3 honest negative has multiple causes:
   - CFM path-(a) GPU BLOCKED per `wf_gpu_recovery_now` 2026-09-15
   - 3-layer singleton attractor per `WF-Lambda-Internal-Review` 2026-09-15
   - search-bound `n_simulations=100` per `WF-Round12-Lambda-Pilot` 2026-09-15
   
   The 5 tasks here attack the **middle layer** (MCTS branch coverage, diversity axis, prior variance, dead-seed budget) but cannot fix GPU CFM or lift the hard simulation cap.

5. **CFM path-(a) lift remains BLOCKED by GPU outage** per `wf_gpu_recovery_now` 2026-09-15 (cuda_available=True now, but decode_ratio=0/192 → FAILURE per spec gate; path (c) λ-only stays as default). The 5 Phase-2 tasks do not address this.

6. **Disjoint file-set guarantee is exact**, not approximate. The `line_overlap_check` field in `phase2_task_plan.json` documents that T1 edits `_expand` inner loop at lines 2903-2919 and T3 edits the pre-loop cache-hit branch at lines 2898-2902 — verified non-overlapping.

7. **All algorithms lit-grounded + math-prior**:
   - T1: Auger 2013 §3.2 (DPW) — algebraic bound
   - T2: Bemis 1996 (scaffold algebra) + Polykovskiy 2020 (IntDiv set-Jaccard)
   - T3: Auger 2013 Thm 1 + closure_theorem.tex O(B·N·|R|) bound
   - T4: Neu 2017 (entropy-MDP regularisation) + Williams 1992 (variance reduction bound)
   - T5: PEP 387 (no math change)

---

## 7. Files (absolute paths)

- Phase 1A lit: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_algo_tune/phase1a_lit_research.md`
- Phase 1B code review: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_algo_tune/phase1b_code_review.md`
- Phase 2 task plan JSON: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_algo_tune/phase2_task_plan.json`
- Phase 2 summary (this file): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_algo_tune/phase2_summary.md`

End of Phase 2.
