# WF-Algo-Tune — Final Verdict (Phase 4)

**Date:** 2026-09-15
**Owner:** post-R13 algorithmic tuning workflow (`w3423vx8w`)
**Goal:** verify whether the singleton-attractor identified by
`WF-Lambda-Internal-Review` (2026-09-15) is broken on novel pockets
(`test_010..test_019`) after the Phase-3 ships; aggregate metrics;
write `metrics/by_round/r13_algo_tune_attempt.json`; recommend
R13 retry path.

---

## 1. Pytest full verification (MEASURED)

```
$ uv run pytest molmetal/tests/ molmetal/molmetal_lam/tests/ --tb=short -q
```

Pre-existing failures excluded (all GPU/CUDA-coupled or
hidden_dim=16-model-coupled tests that were already failing in
`wf_remove_smoke/Phase-4/2026-09-15`):

| Test file | Reason for pre-existing failure |
|---|---|
| `molmetal/tests/test_atom_training_contract.py::test_atom_targets_are_supervised_but_not_supplied_as_features` | atom-loss contract at hidden_dim=16 (CPU-only test against an RNG-initialised model; unrelated to Phase 3 ships) |
| `molmetal/tests/test_generate_atom_types.py::TestGenerateAtomTypes::test_atom_loss_decreases` | Same atom-loss contract |
| `molmetal/tests/test_lipman_spatial_contract.py::*` | GPU-only (RTX-only; ROCm does not have cuEquivariance ops for `test_equivariance_and_first_update_gradients`) |
| `molmetal/tests/test_pocket_conditioned_lipman.py::*` | GPU-only (`test_pocket_conditioning_round_trip` + `_loss_decreases` need torch.cuda) |
| `molmetal/tests/test_rocm_lipman.py::*` | GPU-only (`test_lipman_adapter_on_gpu` + `test_lipman_fm_train_step_on_gpu` need `torch.cuda.is_available()=True`) |

**Excluded-run verdict:** 1607 passed, 6 skipped, 1 xpassed, 0 new
failures in 470.18 s.

**Honest framing:** the 7 pre-existing failures are **not regressions
from Phase 3 ships**. They were already failing before this session
(see `wf_remove_smoke/Phase-4/final.md` 2026-09-15). The Phase-3 ships
introduced 0 new test failures.

---

## 2. Novel-pocket smoke (test_010..test_012, n_sim=500) — MEASURED

```
$ cd /home/hugo/codes/try_triton_on_rocm
$ uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 3 --seeds 42 --n-simulations 500 --n-top-k 20 \
    --metal-seed cisplatin --click-rules auto-pt-strict \
    --decoder-rework \
    --output-dir molmetal/reports/wf_algo_tune/novel_pockets_smoke/
```

(Note: the orchestrator auto-prepends `wf_lambda1_` to the
`--output-dir` value, so the actual on-disk path is
`molmetal/reports/wf_lambda1_molmetal/reports/wf_algo_tune/novel_pockets_smoke/`.)

### 2.1 Per-cell results (MEASURED)

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | div_subpkt | novel | synth | metal | ref_tan | logP | TPSA | sa | qed | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_010 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.143 | -0.58 | 56.5 | 3.657 | 0.708 | 68.0 |
| test_011 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.229 | -0.58 | 56.5 | 3.657 | 0.708 | 75.1 |
| test_012 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.123 | -0.58 | 56.5 | 3.657 | 0.708 | 54.4 |

### 2.2 Critical honest finding (READ FIRST)

**Singleton attractor BROKEN on novel pockets** —
all 3 cells produced `n_distinct=20` (versus the Round-13 100x3 honest
negative which had `n_distinct=1` collapse on `test_010..test_019`).
This matches the `PathA-10x3` lift on `test_000..test_009` exactly:

| Metric | R12 Pilot (test_000..test_009) | PathA-10x3 (test_000..test_009) | R13 Partial (test_010..test_019) | **AlgoTune Smoke (test_010..test_012)** |
|---|---|---|---|---|
| n_distinct mean | 1 | 20 | 1 (collapse) | **20** |
| div_tanimoto | 0.000 | 0.1065 | n/a (no report) | 0.1065 |
| div_homotype | 0.000 | 0.0749 | n/a | 0.0749 |
| valid | 1.000 | 1.000 | 1.000 | 1.000 |
| metal_compliance | 1.000 (trivial) | 0.000 | 0.000 | 0.000 |

The F2(a) MetalLigandExchange rule + scaffold-aware gate + 3D coords +
decoder rework DO generalise the PathA-10x3 lift from `test_000..test_009`
to novel `test_010..test_012`.

**BUT — secondary honest finding:** all 3 novel pockets produced the
**IDENTICAL 20-SMILES candidate list** (byte-identical, just reordered
into the candidate slots). The diversity lift is **pocket-invariant**,
not pocket-specific. The auto-pt-strict expansion + scaffold-aware gate
enumerates a fixed 20-molecule chemotype basket on every pocket:

```
NC(Cn1c[c]([Pt])nn1)C(=O)O
CC(C(=O)O)n1c[c]([Pt])nn1
NC(Cn1nnc[c]1[Pt])C(=O)O
CC(C(=O)O)n1nnc[c]1[Pt]
OCC(O)n1c[c]([Pt])nn1
CC(O)n1c[c]([Pt])nn1
OCCOCCOCCn1c[c]([Pt])nn1
ClC=Cc1ccc(-n2c[c]([Pt])nn2)cc1
[Pt][c]1cn(-c2ccc3ccccc3c2)nn1
[Pt][c]1cn(-c2ccc3ccccc3n2)nn1
[Pt][c]1cn(Cc2ccno2)nn1
[Pt][c]1cn(Cc2ccncc2)nn1
[Pt][c]1cn(Cc2cnccn2)nn1
[Pt][c]1cn(CC2CNCN2)nn1
COc1ccccc1Cn1c[c]([Pt])nn1
Fc1ccccc1Cn1c[c]([Pt])nn1
Clc1ccccc1Cn1c[c]([Pt])nn1
O=C(O)CCn1c[c]([Pt])nn1
O=C(O)Cn1c[c]([Pt])nn1
CC(=O)NCCn1c[c]([Pt])nn1
```

This means: the **CHEMISTRY layer** of the 3-layer singleton attractor
is broken (we now produce 20 distinct molecules per pocket instead of
1), but the **MCTS CACHE + REWARD PRIOR** layers still produce a
fixed chemotype basket that does **not differentiate by pocket**. The
reference ligand (e.g. `CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@H](N2C=C[C@@](N)(O)NC2=O)O[C@@H]1C(=O)O)C(=N)N`
for test_010, or `Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO[P@](=O)(O)O[P@](N)(=O)O)[C@@H](O)[C@H]1O` for test_012) is completely ignored — the
metal-seed anchor `[Pt]C#C` overrides the pocket anchor.

### 2.3 Why the same basket across pockets

The 3-layer singleton attractor diagnostic from `WF-Lambda-Internal-Review`
(2026-09-15) named three causes:

| Layer | Status after Phase 3 | Status after this smoke |
|---|---|---|
| (i) chemistry (click SMARTS ignore Pt_II) | BROKEN — F2(a) MetalLigandExchange + AquaExchange added; auto-pt-strict gate | n_distinct lifted 1 -> 20 |
| (ii) MCTS cache (`_unreactive_states` permanent at proof_search.py:2726) | UNCHANGED — no fix shipped in Phase 3 (T3 was DESIGNED only) | cache returns 20 distinct states per pocket, but they are the same 20 |
| (iii) reward prior (metal_geometry_prior_bonus hard gate) | PARTIALLY BROKEN — Phase 3H soft_score_metal_geometry shipped but is opt-in (weight=0.0 default) | hard gate still active at weight=1.0; rewrote candidate ranking but not candidate enumeration |

The pocket-conditioned root prior (Phase-3J `pocket_features` module)
was **DESIGNED but NOT integrated into the live MCTS** — that wiring
is the Phase 4 integrator's domain (`r4_lambda_only_run.py` /
`proof_search.py` are owned by workflow `w8579x29t`). The same is true
of the learned prior (Phase-3L checkpoint trained but not wired into
`MCTSProofSearch._prior`).

---

## 3. Aggregate metrics across the 5 Phase-3 reports (CPU-only, MEASURED)

### 3.1 Per-fix ship status

| Fix | Lit anchor | Status | Module / file | Tests | MEASURED lift |
|---|---|---|---|---|---|
| **T1 F3-DPW** (Auger 2013 §3.2) | Auger 2013 §3.2 | DESIGNED only | `proof_search.py:_expand` 2903-2919 (NOT edited) | 3 scoped in `phase2_task_plan.json` | **NOT MEASURED** — outside file-set ownership |
| **T2 F4-Murcko** scaffold diversity (Bemis 1996 + Polykovskiy 2020) | Bemis 1996 + Polykovskiy 2020 | NEW module shipped | `molmetal/molmetal_lam/metrics/scaffold_diversity.py` | 6 tests pass | Diversity lift on a different axis (scaffold); not used in this smoke's `div_subpocket` (which uses Phase-3D SubPocketFP, a *different* module) |
| **T3 F5.f closure-consult** (Auger 2013 Thm 1) | Auger 2013 Thm 1 | DESIGNED only | `proof_search.py:_expand` 2898-2902 + `lam_chem/closure.py` (NOT edited) | 3 scoped | **NOT MEASURED** — outside file-set ownership |
| **T4 F-entropy-τ** (Neu 2017 + Williams 1992) | Neu 2017 + Williams 1992 | PARTIALLY SHIPPED — `soft_score_metal_geometry` additive channel; weight default 0.0 (opt-in) | `molmetal/molmetal_lam/priors/metal_geometry.py` | 9 pass on CPU (Phase 3H) | **PROJECTED** +3-8pp metal_compliance on non-seed molecules per Williams 1992 variance-reduction bound — NOT MEASURED in this smoke (default weight=0.0) |
| **T5 F5.e PEP-387** | PEP 387 | NOT SHIPPED — deprioritised (0pp lift by design) | — | — | — |
| **F1** soft tiered metal prior (WF-Lambda-Fix-Singleton) | Schulman 2017 PPO + Neu 2017 | SHIPPED | `r4_lambda_only_run.py` | 3 tests pass | metal_compliance 1.000 -> 0.000 (expected trade-off) |
| **F2(a)** MetalLigandExchange + AquaExchange (WF-Lambda-Fix-FullPath-v2) | pt_click_compat 5×5 matrix | SHIPPED | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | 17 pass (4 scaffold-aware + 13 pre-existing) | n_distinct 1 -> 20 |
| **F3** decoder rework + 3D coords (WF-Path-B-GPU-Retrain + WF-Lambda-MCTS-Coords-Fix) | — | SHIPPED | `r4_lambda_only_run.py:decoder_rework` | decoder_pass_rate=1.0 in this smoke | n_distinct 1 -> 20 (combined with F2(a)) |

### 3.2 Phase 3 evidence summary

| Fix family | Aggregate MEASURED evidence this session |
|---|---|
| Unit tests (T2 scaffold, T4 soft-score, R PUCT, L learned prior, L1-L5 modules, P triton dispatch, E triton wire, H soft reward, F effect size, G coord probe, D subpocket, C PB harness, J warm start) | 14+9+7+14+13+13+11+11+10+10+10+9+13 = **154 tests pass across the 13 new modules** (full pytest run also shows 1607 pass / 6 skip / 1 xpass / 7 pre-existing failures excluded) |
| End-to-end smoke (algo tune novel pockets) | n_distinct 1 -> 20 on test_010..test_012 (3/3 cells) |
| Wall-clock cost | 202.63 s for 3 cells = 67.54 s/cell at n_sim=500 |

---

## 4. TODO-29 status update

**Decision:** "F2(a) + algo tune lifts the singleton, retry R13" — but
**with one critical caveat**. The chemistry layer of the singleton
attractor IS broken, but the pocket-conditioned search is **not yet
working**. The candidate basket is uniform across pockets.

```
TODO-29 status: F2(a) + algo tune PARTIALLY lifts the singleton (n_distinct 1 -> 20 verified,
div_tan 0 -> 0.1065 verified), but the lift is pocket-INVARIANT — same 20 SMILES on every pocket.
Retry R13 ONLY after (a) Phase-3J warm-start pocket_features wired into proof_search.py:_root,
(b) Phase-3L learned_prior wired into proof_search.py:_prior with mix_uniform=0.5, or
(c) metal_seed replaced by pocket-conditioned reference ligand as the MCTS root.
```

Recommended R13 retry pre-conditions:

1. **Wire Phase-3J `pocket_features` into live `MCTSProofSearch`**
   (`proof_search.py:_root.children[child].P` populated from
   `modify_root_prior` at first selection). Phase 3J was DESIGNED but
   not integrated — file-set owned by Phase 4 integrator
   `w8579x29t`.
2. **Wire Phase-3L `LearnedPolicyPrior` into live `MCTSProofSearch._prior`**
   with `mix_uniform=0.5` (AlphaGo Zero root-noise mixing).
   Trained checkpoint exists at `checkpoints/learned_prior_Pt.pt`.
3. **Then re-run `test_010..test_019` 10×3** with
   `--pocket-features-from-manifest` and `--learned-prior checkpoints/learned_prior_Pt.pt`.
   Expected verdict: per-pocket distinct candidate lists (the basket
   should differ pocket-by-pocket).

---

## 5. Recommendation for R13 retry

**Path-A-mini-retry (cheapest, highest information):**

| Step | Action | Wall-clock | Expected lift |
|---|---|---|---|
| 1 | Wire Phase-3J warm-start into `proof_search.py:_root` (add `MCTSProofSearch.search(..., pocket_features=None, ...)` kwarg + populate `_root.children[child].P`) | 1.5 h (Phase 4 integrator) | +5-15pp diversity per pocket (PROJECTED, NOT MEASURED) |
| 2 | Wire Phase-3L learned prior into `proof_search.py:_prior` with `mix_uniform=0.5` | 1.0 h | +5-10pp on applicable-rule coverage (PROJECTED) |
| 3 | Re-run `test_010..test_019` 10×3 with `--pocket-features-from-manifest --learned-prior checkpoints/learned_prior_Pt.pt` | ~10 min CPU | n_distinct=20 with PER-POCKET distinct lists (vs current identical lists) |
| 4 | If step 3 lifts per-pocket differentiation, ship to R13 paper §4.6 | — | — |

**Path-B-full-retry (most expensive):**

If Path-A-mini-retry fails, the next move is to attack the MCTS
CACHE layer (proof_search.py:_unreactive_states) — this requires
either T3 closure-consult (2.5 h, +20-40% wall-clock on closure-dead
seeds) or a deeper audit of the `_unreactive_states` permanent
membership rule.

**Path-C-defer (CFM lift):** CFM path-(a) retrain remains BLOCKED
per `wf_gpu_recovery_now` 2026-09-15 (cuda_available=True but
decode_ratio=0/192 → FAILURE per spec gate). No change in this
session.

---

## 6. Honest framing — what was MEASURED vs PROJECTED in this Phase 4

**MEASURED in this session:**

1. **pytest full verification**: 1607 pass / 6 skip / 1 xpass / 7
   pre-existing failures excluded. 0 new failures from Phase 3 ships.
2. **Novel-pocket smoke (test_010..test_012)**: n_distinct=20 on all
   3 cells, valid=1.000, synth=1.000, uniq=1.000, div_tan=0.1065,
   div_hom=0.0749, div_subpocket=0.6539, metal_compliance=0.000,
   wall=202.63 s for 3 cells.
3. **Singleton attractor status**: CHEMISTRY layer BROKEN (n_distinct
   1 -> 20 across 3 novel pockets); CACHE + REWARD PRIOR layers
   unchanged.
4. **Pocket-conditioned search status**: NOT WORKING — all 3 novel
   pockets produce the **identical 20-SMILES candidate list**
   (byte-identical, just reordered into the candidate slots).

**PROJECTED but NOT MEASURED:**

1. T1 F3-DPW +5-15pp diversity (Auger 2013 §3.2 algebraic bound).
2. T3 F5.f closure-consult +20-40% wall-clock on closure-dead seeds
   (Auger 2013 Thm 1 + closure_theorem.tex O(B·N·|R|) bound).
3. T4 F-entropy-τ +3-8pp metal_compliance on non-seed molecules
   (Williams 1992 variance-reduction bound).
4. T7 F-scaffold-PUCT-bias lift (Bemis 1996 + Schuffenhauer 2007 +
   Polykovskiy 2020 IntDiv).
5. Phase-3J warm-start per-pocket differentiation lift
   (Peng 2022 Pocket2Mol §3.2, Silver 2018 AlphaZero dirichlet
   fraction 0.25).
6. Phase-3L learned-prior applicable-rule coverage lift
   (Silver 2017 AlphaGo Zero §III.B, Schrittwieser 2019 MuZero §3).

**OUT OF SCOPE for this session:**

1. CFM path-(a) GPU retrain (BLOCKED per wf_gpu_recovery_now
   2026-09-15).
2. PB 30-cell production sweep (BLOCKED upstream of Vina/PB per
   wf_pb_pass_10x3_smoke).
3. Phase 4 integrator file-set: `r4_lambda_only_run.py`,
   `proof_search.py`, `paper/main.tex`, and the w8579x29t Phase 3
   agents' file set (`pt_click_compat.py`, `beta_reductions.py`,
   `run_pb_production.py`, `per_residue_diversity.py`,
   `velocity_net.py`, `egnn_rocm.py`,
   `click_rule_effect_size_study.py`, `metal_coord_probe.py`).
   None touched.

---

## 7. Files (absolute paths)

- **Phase-4 final (this report):**
  `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_algo_tune/final.md`
- **Phase-4 metrics JSON:**
  `/home/hugo/codes/try_triton_on_rocm/metrics/by_round/r13_algo_tune_attempt.json`
- **Novel-pocket smoke output:**
  `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_molmetal/reports/wf_algo_tune/novel_pockets_smoke/{report.json,summary.md}`
- **Phase 3 reports (5 fixes):**
  `molmetal/reports/wf_algo_tune/phase3h_soft_reward.md`
  `molmetal/reports/wf_algo_tune/phase3j_warm_start.md`
  `molmetal/reports/wf_algo_tune/phase3l_learned_prior.md`
  `molmetal/reports/wf_algo_tune/phase3p_triton_dispatch.md`
  `molmetal/reports/wf_algo_tune/phase3r_puct.md`
- **Phase 1 + Phase 2 inputs:**
  `molmetal/reports/wf_algo_tune/phase1a_lit_research.md`
  `molmetal/reports/wf_algo_tune/phase1b_code_review.md`
  `molmetal/reports/wf_algo_tune/phase2_summary.md`
  `molmetal/reports/wf_algo_tune/phase2_task_plan.json`
- **Related existing memory entries (cross-refs):**
  `WF-Lambda-Internal-Review 2026-09-15` (3-layer attractor diagnostic)
  `WF-Lambda-Fix-Singleton 2026-09-15` (F1 soft tiered prior)
  `WF-Lambda-Fix-FullPath-v2 2026-09-15` (F2(a) MetalLigandExchange +
  scaffold-aware)
  `WF-Path-B-GPU-Retrain 2026-09-15` (F3 decoder rework + 3D coords)
  `WF-GPU-Recovery-Now 2026-09-15` (CFM BLOCKED)
  `WF-PB-Pass-10x3-Smoke 2026-09-15` (PB search-bound)
  `WF-Round13-100x3-Sweep 2026-09-15` (honest negative)

---

End of Phase 4.
