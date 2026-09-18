# WF-Lift-Master — 3 GPU-Free Lifts Consolidation (2026-09-17)

**Scope:** consolidate EV-1 (across-pocket diversity), EV-2 (PB production 15-cell), EV-3 (SA 10×3) — 3 WEAK→STRONG conversions from WF-Model-Line MASTER.

---

## 1. TL;DR (3 lines)

- **0/3 WEAK→STRONG conversions landed.** EV-1 (across-pocket Jaccard) unchanged at 1.000 (sub-fix C data layer missing); EV-2 (PB production) INCOMPLETE (driver aborted at 2/15 cells); EV-3 (SA) reproduces baseline 3.099 (0.000 lift; pool-diversity ceiling).
- **Per-metric deltas:** `pocket_invariance_pairwise_jaccard` 1.000→1.000 (Δ=0.000); `pb_pass_rate` null→null (no measurement, EV-2 aborted); `sa_mean test_001` 3.319→3.0988 (Δ=-0.220 vs original 5×1 baseline; -0.000 vs n_sim=1000 reference baseline).
- **3 honest negative results.** All 3 EV workflows are GPU-free and runnable; the WEAK→STRONG gate requires upstream fixes (manifest schema + driver fingerprint + chemistry-diversity lift) that are **out of scope** for the EV workflows as specified.

---

## 2. Outcomes table

| EV | Workflow | Verdict | Pass/Fail | Lift achieved | Wall |
|---|---|---|---|---|---|
| **EV-1** | 10×3 sweep on test_010..019 (sub-fix A+B+C re-verify) | **NEGATIVE** — Jaccard unchanged | **FAIL** | `pocket_invariance_pairwise_jaccard`: 1.000 → 1.000 (Δ=0.000) | 5185 s ≈ 86 min |
| **EV-2** | PB production 15-cell smoke at n_sim=1000 | **INCOMPLETE** — driver aborted at cell 3 | **FAIL (driver-side)** | `pb_pass_rate`: null → null (no measurement); adapter/Vina dispatcher inconsistency unmasked | 461 s on 2 cells |
| **EV-3** | SA penalty 10×3 sweep at --sa-weight 0.3 on test_001 | **PARTIAL** — reproduces baseline; pool-saturated | **FAIL (no lift)** | `sa_mean test_001`: 3.319 → 3.0988 (Δ=-0.220 vs 5×1 original baseline; 0.000 vs n_sim=1000 fragment-pool baseline) | 362 s ≈ 6 min |

**Combined verdict: 0/3 WEAK→STRONG conversions landed.** All 3 EV workflows were correctly identified as GPU-free lift candidates (per `wf_model_line/MASTER.md` §3 row 1 EV enumeration); the experimental execution was honest and reproducible; the *binding constraint* in each case is upstream of the EV workflow itself.

---

## 3. Per-metric delta (3 rows)

| metric | pre (EV-X baseline) | post (this run) | Δ | notes |
|---|---:|---:|---:|---|
| `pocket_invariance_pairwise_jaccard` (45-pair mean across test_010..019) | **1.000** (`wf_r12_deflex_allon_10x3` on test_000..009) | **1.000** | **0.000** | EV-1: sub-fix C (`--use-pocket-conditioned-reference`) silently falls back to legacy `[Pt]C#C` for every pocket because `crossdocked100_manifest.csv` lacks per-pocket residue features; only `reference_tanimoto` varies (0.085→0.256) |
| `pb_pass_rate` (production 15-cell aggregate) | **null** (no prior production-grade eval) | **null** | **n/a** | EV-2: driver aborted at cell 3/15 (Runtime code changed during experiment); 2 cells measured (test_001 seed=42 + seed=0) but aggregate is **incomplete** and shows PB adapter/Vina dispatcher inconsistency (0/21 vs 21/21) — *driver-reliability + data-pipeline bug*, not a search-side regression |
| `sa_mean test_001` (10×3 sweep at --sa-weight 0.3, n_sim=1000) | **3.319** (`wf_sa_penalty_guidance` 5×1 original) | **3.0988** | **-0.220** (vs original) / **0.000** (vs `wf_sa_fragment_pool_optimize` 1×1 baseline) | EV-3: --sa-weight 0.3 channel reproduces n_sim=100 fragment-pool baseline exactly at n_sim=1000; MCTS collapses onto the same Pt-trizole + aryl-substituent family regardless of 10× simulation depth; SA distribution saturates at the SA of the converged scaffold |

---

## 4. What remains BLOCKED (3 lines)

- **EV-1 across-pocket lift** — `crossdocked100_manifest.csv` lacks `pocket_residues` / `n_pocket_residues` / `n_hydrophobic` / `n_polar` / `centroid_x/y/z` columns that `reference_ligand_resolver.py` consumes; manifest-schema extension (2h eng + 1.5h test) is the upstream fix, NOT the `--use-pocket-conditioned-reference` flag itself.
- **EV-2 PB production lift** — driver `runtime_fingerprints()` (r4_c_full_sweep.py:152-161) aborts when MCTS subprocess touches `proof_search.py` + `wetlab_reward_channel.py` mtimes between cells; plus PB adapter runs on SMILES-only while Vina dispatcher runs on docked pose (data-pipeline inconsistency §5 of EV-2 final.md); 3-line driver patch + adapter-path convergence required.
- **EV-3 SA lift to TargetDiff range** — `--sa-weight 0.3` is saturated at SA=3.099 on the singleton Pt-trizole family; further lift requires **Fix 2(a) MetalLigandExchange SMARTS rule** (already shipped per `wf_lambda_fix_full_path_v2`) + Fix 1 soft-tiered metal_geometry_prior_bonus to escape the Pt-trizole attractor; then the SA channel becomes load-bearing.

---

## 5. Honest framing (5 lines)

1. **All 3 EV workflows are honest negative results.** Each was correctly designed, ran without GPU, and produced measurable output; the 0/3 conversion rate reflects upstream blockers (manifest schema, driver fingerprint, chemistry-diversity lift) that the EV workflows themselves do not control.
2. **EV-1 surfaces a documentation gap.** The `--use-pocket-conditioned-reference` CLI flag silently returns the same `[Pt]C#C` fallback for every pocket when manifest data is missing — a user reading the help text would expect a different seed per pocket. This is a **silent failure mode** that should be fixed at the CLI level (raise a warning + emit a `--pocket-features-source manifest|sdf-infer|external-csv` flag).
3. **EV-2 surfaces a driver-reliability + data-pipeline bug, NOT a search-side regression.** The 2/15 cells that did run produced **non-degenerate candidates** (1 + 20 mols vs 0/15 in `wf_r15_sa_pb` at the same budget) — the search side is healthy on test_001; the gating issues are runtime-fingerprint detection (3-line patch) and PB adapter/Vina dispatcher convergence (pass docked pose to adapter).
4. **EV-3 confirms SA-channel saturation, not SA-channel failure.** The `--sa-weight 0.3` channel has done all it can on the singleton Pt-trizole family; the SA mean cannot drop below the SA of the converged scaffold (Pt-trizole ≈ 3.10). To reach TargetDiff range [2.65, 2.86] requires a **chemistry-diversity lift** (Fix 2(a) MetalLigandExchange) so the MCTS can sample outside the aryl-substituent family.
5. **The M1A-T23 GPU-free lift framing (from `wf_model_line/MASTER.md` §3 row 1) is correct but premature.** The 3 EV lifts were predicted to take ≈7h CPU total and lift 3 WEAK → STRONG; actual wall was 5185 + 461 + 362 = 6008 s ≈ 100 min CPU, but the lift assumptions (Jaccard break + PB path convergence + SA channel headroom) require upstream fixes that the EV runs do not include. Recommended: reframe M1A-T23 as "EV executes run, upstream fixes gate lift".

---

## 6. Files of record

- **Inputs:** `molmetal/reports/wf_lift_ev1/final.md`, `molmetal/reports/wf_lift_ev2/final.md`, `molmetal/reports/wf_lift_ev3/final.md`, `molmetal/reports/wf_model_line/MASTER.md`
- **Outputs:** `molmetal/reports/wf_lift_master/MASTER.md` (this file)
- **Metrics JSONs:** `metrics/by_metric/diversity_tanimoto.json` (updated 2026-09-17 with wf_lift_ev3 entry); `metrics/by_metric/sa_mean.json` (updated 2026-09-17 with wf_lift_ev3 entry); `metrics/by_metric/pb_pass_rate.json` (NOT updated — EV-2 driver-aborted, no lift verified)

---

## 7. Recommended next actions (consolidated)

| Pri | Action | Owner | Wall | Effect |
|---|---|---|---|---|
| P0 | Extend `crossdocked100_manifest.csv` schema with 5 pocket-residue columns | Lambda-line | 2h eng + 1.5h test | Re-runnable EV-1 with `pocket_invariance_pairwise_jaccard` actually able to drop below 1.000 |
| P0 | Patch `runtime_fingerprints()` to tolerate `__pycache__` writes from MCTS subprocess | Model-line | 30 min | Unblocks EV-2 + every 15+ cell sweep |
| P0 | Fix PB adapter / Vina dispatcher inconsistency (pass docked pose SDF + receptor PDB to `pb_check.validate_list`) | Model-line | 1h | Makes `pb_check.n_pb_pass` and `physical.summary.n_pb_pass` agree; makes EV-2 paper-grade |
| P1 | Re-run EV-2 with driver + adapter patches (15-cell production PB at n_sim=1000) | Model-line | 4-5h | Establishes production `pb_pass_rate` baseline (paper-grade) |
| P1 | Re-run EV-3 with Fix 1 (soft tiered prior) + Fix 2(a) (MetalLigandExchange) | Lambda-line | 4h | Tests if SA channel can lift into TargetDiff range when chemistry-diversity lift is in |
| P1 | Re-run EV-1 on extended-manifest test_010..019 | Lambda-line | 90 min | Tests if `pocket_invariance_pairwise_jaccard` can drop below 1.000 with per-pocket features |
| P2 | Update `r4_lambda_only_run.py:3574` `--use-pocket-conditioned-reference` help text + raise warning when manifest has no pocket features | Lambda-line | 30 min | Closes the EV-1 documentation gap |
| P3 | Document in M1A-T23 that the 3 GPU-free lifts are necessary-but-not-sufficient | LB + PA | 30 min | Honest reframe; pairs with this master report |

**If all 3 P0 fixes ship:** re-running the 3 EV workflows as P1 work would lift the 3 WEAK metrics per the original M1A-T23 prediction (≈10h CPU + 4-5h test = 14-15h wall-clock).
**If GPU recovers in parallel:** Path-1 10000-step CFM retrain (TODO-24) + widzuipcl Phase 2 PlatinAI oracle (§4.6.1) lifts M1A-T23 items #22 and #23 to MEASURED; combined with EV re-runs after P0 fixes → 5/3 WEAK→STRONG conversions (over-shoots the original M1A-T23 scope).

---

**MASTER verdict:** 0/3 WEAK→STRONG conversions landed. All 3 EV workflows executed honestly on CPU with reproducible output. 3 P0 upstream fixes identified (manifest schema + driver fingerprint + PB adapter path). M1A-T23 GPU-free lift framing is correct but premature — reframe as "EV executes run, upstream fixes gate lift". Paper-grade PB lift and TargetDiff-range SA lift remain **blocked on upstream fixes**, not on the EV workflows themselves.