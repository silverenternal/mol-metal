# Wf-327-F16 — m7 Diversity Multi-PH mechanical lift investigation

**Status:** ⚙️ **INVESTIGATED — 2026-09-29**
**Wf-325G status:** VERIFIED (was designed + script authored but never executed; this Wf ran the script + characterised the failure mode)
**Owner:** Hugo Lee
**Depends on:** Wf-325G (script `molmetal/scripts/wf325g_m7_multi_ph_lift.py`) + R12 EV-3 baseline `metrics/by_metric/diversity_tanimoto.json`
**Honors user directive (2026-09-29):** "感觉现在好像还不是启动重新训练的最佳时机，得想办法先把 honest-negative 先解决了" — this investigation is **CPU-only**, **0 GPU launches**, **0 retrain**, **does not interrupt any in-flight training**.

---

## §0 One-line answer

**m7 Diversity 0.1366 is NOT pool-bottlenecked — it is MCTS-bottlenecked.** The 449-pool ceiling is 0.9068 (MEASURED, this run); the generated-cohort at 0.1366 collapses because the MCTS rollout funnels to similar molecules even from a diverse pool. Multi-PH MaxMin selection is a viable **post-rollout diversity filter** candidate but the Wf-325G script's "PASS" verdict is an apples-to-oranges artifact (pool-cohort vs generated-cohort comparison). Root cause: **METRIC_DEFINITION + MCTS_EXPLORATION_BOTTLENECK**, NOT LIBRARY_TOO_SMALL.

---

## §1 m7 Diversity: what is it actually measuring?

**Source:** `metrics/by_metric/diversity_tanimoto.json` (cited verbatim).

| Property | Definition |
|---|---|
| **Metric** | `diversity_tanimoto` |
| **Formula** | Mean pairwise Tanimoto **distance** (1 − similarity) on Morgan-ECFP4 fingerprints (radius=2, n_bits=2048) |
| **Scope** | **Generated cohort** (Lambda MCTS path output), NOT pool diversity, NOT inter-pocket |
| **Unit** | ratio (0-1, higher = more diverse) |
| **MEASURED value (R12 Path A, 10×3 sweep)** | **0.1065** (deterministic per-cell std=0.0; n_distinct_mean=20 of 30) |
| **MEASURED value (R12 EV-3, 10×3 sweep, +sa-weight 0.3)** | **0.1366** (+28% lift over Path A) |
| **MEASURED value (TODO-180/Wf-180, 30-cell 3-seed, +Tanimoto reward)** | **0.0397** (different cohort — paper-facing number) |
| **lit TargetDiff (Guan 2023 ICLR) ceiling** | 0.860 |
| **Gap to ceiling** | −0.7234 (honest per `party-twenty-spirit` §3.5) |

**What it does NOT measure:**
- Pool diversity (the 449-pool ceiling is 0.9068 — see §3)
- Inter-pocket diversity (cross-pocket pIC50 spread is m21 territory)
- Diversity-subpocket (per-residue 8 Å Morgan-ECFP4 — see `diversity_subpocket.json`, DESIGN_ONLY per Wf-318G)

**Key insight:** m7 measures what the MCTS **generates**, not what the pool **contains**. This distinction is the root of the entire honest-negative.

---

## §2 Wf-325G status verification

| Item | Status |
|---|---|
| Script `molmetal/scripts/wf325g_m7_multi_ph_lift.py` | **SHIPPED** (2026-09-29, 684 LOC) |
| Report dir `molmetal/reports/wf325g_m7_multi_ph_lift/` | **EXISTED but EMPTY** (0 files) before this run |
| TODO/completed/325g_m7_diversity.md | **NEVER AUTHORED** before this run |
| Pre-flight gate for Wf-325G result.json | **NOT in pre_flight_check.py v2 25/25 PASS suite** (deferred; per-script file-presence only) |
| **This run** | EXECUTED (2026-09-29T04:27:10Z); result.json + per_cohort.csv written |

**Conclusion:** Wf-325G was **designed but never executed** prior to this Wf. The script was authored as a CPU-only scaffold to test whether Multi-PH integration at the MCTS-heuristic level could mechanically lift m7 above R12 EV-3 (0.1366). This run is the first execution.

---

## §3 Wf-325G measurement (this run, 50 pools × 20 SMILES × 4 selected = 50 cohorts)

Source: `molmetal/reports/wf325g_m7_multi_ph_lift/result.json`

| Metric | Value | Source |
|---|---:|---|
| n_smiles_loaded | **449** | `metallo_drugs_500_train.csv` (96 platinai + 19 metal_cytotox + 334 tmqm) |
| Multi-PH selector availability | **OK** | `_build_default_signature()` returned `MultiPHPLLSignature` |
| **m7 pool ceiling** (full 449-pool) | **0.906782** | MEASURED (Morgan-ECFP4 pairwise Tanimoto distance, full pool) |
| m7 generated-baseline this run (50 raw pools of 20) | **0.904122** | MEASURED (mean pool diversity before selection) |
| **m7 after Multi-PH MaxMin selection** (50 pools × 4 selected each, aggregated dedup) | **0.932428** | MEASURED (50 cohort aggregation; n_aggregated_candidates=17 due to dedup) |
| Δ vs generated-baseline this run | **+0.028307** | REAL — same cohort type, apples-to-apples |
| Δ vs R12 EV-3 reference (0.1366) | **+0.795828** | **ARTIFACT — apples-to-oranges** (pool-cohort vs MCTS-generated-cohort) |
| Per-pool lift (n=50) | mean=**+0.0195**, min=**−0.0690**, max=**+0.0980** | MEASURED — 30/50 pools with positive lift |
| Multi-PH signature success rate | **0.79** (316/400) | MEASURED — most metallo SMILES DO embed |
| Wall time | **442.7 s selection + 0.3 s baseline = ~7.4 min** | CPU-only |

### §3.1 Honest correction to script docstring

The script's docstring claimed **"Multi-PH signature compute FAILS on heavy-metal coordination complexes (Sc2+3, Pt2+, etc.) because RDKit AllChem.EmbedMolecule cannot embed those SMILES"**. This is **INCORRECT** for the metallo cohort we tested:

- 316/400 = 79% of Multi-PH signature computes succeeded on the metallo cohort
- The 84 failures (21%) are graceful fallback to Tanimoto distance
- The `multiph_embeds_fail_on_metallo_drugs` flag in `honest_framing` was `false` per the actual measurement (script default was `multiph_embeds_fail_on_metallo_drugs: multi_ph_success_rate < 0.5`)

**Note:** the script's pessimistic honest-negative about Multi-PH-on-metallo was based on prior cohort assumptions that did not hold here.

### §3.2 The "verdict: PASS" is misleading

The script computes `verdict = PASS` if `delta_vs_r12_ev3 >= 0.05`. Our Δ vs R12 EV-3 is +0.7958 — but this comparison is **apples-to-oranges**:

| Cohort type | What it captures | What it does NOT capture |
|---|---|---|
| **Wf-325G pool cohort** (this run) | Pool-cohort diversity after random sampling + Multi-PH MaxMin selection | MCTS path collapse |
| **R12 EV-3** (baseline) | MCTS-generated cohort diversity on the SAME pool | Pool ceiling (would be 0.907 if MCTS explored fully) |

The +0.7958 "lift" is mostly **comparison artifact**, not a real MCTS-path improvement. The honest measurement is **Δ vs generated-baseline this run = +0.0283** — Multi-PH MaxMin selection provides a +2.8% lift over random sampling from the same pool.

---

## §4 Root cause categorization

Per the task brief, the root cause is one of: `LIBRARY_TOO_SMALL` / `SEED_BIAS` / `METRIC_DEFINITION` / `HONEST_NEGATIVE_BLOCKED`.

### §4.1 LIBRARY_TOO_SMALL? **FALSE**

The 449-pool ceiling is **0.9068** (MEASURED, this run). The pool is already diverse — the bottleneck is downstream of the pool, in the MCTS path that **generates** from it.

**Previous assumption (incorrect):** "The 0.1065 pool-diversity ceiling is structural on the current fragment pool" (per `TODO/completed/82_metric_gap_filling.md` §G4). The 0.1065 is **NOT** a pool-diversity ceiling — it is a **generated-cohort** diversity on the SAME pool.

**Implication:** The TODO-25 P2.3 fragment-pool expansion is **NOT** the right fix for m7 Diversity 0.1366. Pool is already at 0.907. Expanding the pool further will not lift m7 if the MCTS still collapses to similar molecules.

### §4.2 SEED_BIAS? **FALSE**

Per `diversity_tanimoto.json`:
- `cell_std: 0.0` (deterministic per-cell)
- `deterministic_per_cell: true`
- 3 seeds × 10 cells produce identical molecules per cell

This is NOT seed bias — the MCTS deterministically converges to the same molecule regardless of seed. Adding more seeds will not fix this.

### §4.3 METRIC_DEFINITION? **TRUE (PARTIAL)**

m7 measures **generated-cohort diversity**, not pool diversity. The metric is well-defined (Morgan-ECFP4 mean pairwise distance on the MCTS output) but its **interpretation** as a "diversity ceiling" is misleading — the pool ceiling is 0.907, the generated-cohort at 0.1366 is MCTS-bottlenecked, and the gap is **MCTS path collapse**, not a metric definition problem.

**Reframe:** the m7 metric is correctly defined. The honest-negative is on the **MCTS exploration bottleneck**, not on the metric.

### §4.4 HONEST_NEGATIVE_BLOCKED? **TRUE (with CPU-friendly next step)**

Per `party-twenty-spirit` §3.5: "When a metric is BELOW field-avg or BLOCKED on external dependency, declare it as honest-negative — never paraphrase as 'achievable' / 'promising' without PROJECTED tag + wall budget."

The current state:
- m7 0.1366 (MEASURED, R12 EV-3) → honest-negative vs TargetDiff 0.860 ceiling (gap −0.7234)
- Pool ceiling 0.907 (MEASURED, this run) → ceiling is fine
- MCTS path collapse → the structural bottleneck

**The fix is a CPU-only code change to the MCTS rollout heuristic**, not a retrain. Specifically: wire `select_top_n_by_multi_ph_distance` (the Wf-325G MaxMin selector) into `molmetal/molmetal_lam/search_alg/proof_search.py` as a post-rollout diversity filter.

---

## §5 CPU-only fix feasibility

### §5.1 What Wf-325G proved

The Wf-325G CPU-only scaffold proves that:
- Multi-PH signature compute works on **79% of metallo SMILES** (316/400) — prior pessimism was incorrect
- Multi-PH MaxMin selection provides **+2.8% lift** over random sampling on the pool (apples-to-apples)
- The pool ceiling is **0.907** — already diverse
- The +0.7958 vs R12 EV-3 is an apples-to-oranges artifact (pool-cohort vs MCTS-generated-cohort)

### §5.2 Proposed CPU-only next step (no GPU retrain needed)

**Wire `select_top_n_by_multi_ph_distance` into MCTS as a post-rollout diversity filter:**

1. Copy `select_top_n_by_multi_ph_distance` from `molmetal/scripts/wf325g_m7_multi_ph_lift.py` into `molmetal/molmetal_lam/search_alg/proof_search.py` (or a new helper module).
2. Modify the MCTS rollout loop in `proof_search.py` to:
   - Generate N candidate molecules per cell (e.g., N=20 instead of N=1).
   - Apply `select_top_n_by_multi_ph_distance(..., top_n=4)` to pick the most diverse top-4.
   - Return the first one (or all 4 — to be tested).
3. Re-run R12 Path A fixtures (10×3 = 30 cells, 3 seeds) on the **same** cohort.
4. Measure m7 on the **same** generated cohort as R12 EV-3 (apples-to-apples comparison).
5. **Success criterion:** m7 ≥ 0.40 (2.9× lift over 0.1366 baseline). If achieved → multi-seed × multi-pocket rerun is in scope. If not → honest-negative preserved per `party-twenty-spirit` §3.5.

**Wall budget:** CPU-only, ~2 hr dev + ~1 hr test rerun = 3 hr total. No GPU launch. Does not interrupt any in-flight training.

### §5.3 What this CPU fix does NOT do

- Does NOT lift m7 to 0.860 (TargetDiff ceiling) — that requires 7× the current lift, only achievable via a retrain with bigger pool + multi-PH integrated into the loss
- Does NOT fix the structural bottleneck — the MCTS collapse is a deeper algorithmic issue (exploration vs exploitation tradeoff)
- Does NOT claim 0.65 (m7 gate per WEAK_TO_STRONG_PLAN) is reachable without a retrain

---

## §6 Honest framing (per `party-twenty-spirit` §3.5)

**What this Wf DOES claim:**
- Wf-325G script executed successfully, result.json + per_cohort.csv written
- 449-pool ceiling is 0.9068 (MEASURED, this run) — the pool is NOT the bottleneck
- Multi-PH signature compute works on 79% of metallo SMILES (316/400)
- The +0.7958 "lift" vs R12 EV-3 is an apples-to-oranges artifact (pool-cohort vs MCTS-generated-cohort)
- Multi-PH MaxMin selection provides +2.8% real lift over random sampling (apples-to-apples)
- m7 0.1366 → honest-negative vs TargetDiff 0.860 ceiling (gap −0.7234) is preserved
- Root cause is **MCTS EXPLORATION BOTTLENECK**, NOT pool-too-small (refutes the `82_metric_gap_filling.md` §G4 "structural fix = pool expansion" assumption)
- CPU-only fix path is identified (wire Wf-325G MaxMin selector into MCTS dispatch)

**What this Wf does NOT claim:**
- That m7 can reach 0.65 (M7_GATE) without retrain — would require structural MCTS redesign
- That m7 can reach 0.860 (TargetDiff ceiling) without retrain — unrealistic without pool expansion + multi-PH integrated into loss
- That Multi-PH integration alone solves the diversity problem — Multi-PH provides +2.8% pool-selection lift, but the MCTS still collapses to similar molecules
- That the TODO-25 P2.3 fragment-pool expansion is the right fix — pool is already at 0.907 ceiling
- That the Wf-325G script's "verdict: PASS" is honest framing — the PASS verdict is an apples-to-oranges artifact

---

## §7 Open follow-ups

| # | Action | Wall | Owner | Status |
|---|---|---:|---|---|
| 1 | Wire `select_top_n_by_multi_ph_distance` into `proof_search.py` MCTS dispatch | 2 hr | Hugo | TODO (CPU-only, no GPU) |
| 2 | Re-run R12 Path A fixtures (10×3 = 30 cells, 3 seeds) on wired MCTS | 1 hr | Hugo | TODO (CPU-only) |
| 3 | Compare m7 lift (apples-to-apples, R12 EV-3 vs new wired) | 10 min | Hugo | TODO |
| 4 | If lift ≥ +0.20 absolute (m7 ≥ 0.40): multi-seed × multi-pocket rerun | 4 hr CPU | Hugo | CONDITIONAL |
| 5 | Refute the `82_metric_gap_filling.md` §G4 "pool expansion = m7 fix" assumption | doc edit | Hugo | TODO (in this doc) |
| 6 | Decide whether to deprecate TODO-25 P2.3 (fragment-pool expansion) | decision | Hugo | TODO |
| 7 | Update WEAK_TO_STRONG_PLAN §2.5 with new root-cause understanding | doc edit | Hugo | TODO |

---

## §8 File paths

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf325g_m7_multi_ph_lift.py` | NEW Wf-325G CPU scaffold (684 LOC) |
| `molmetal/reports/wf325g_m7_multi_ph_lift/result.json` | NEW Wf-325G measurement result |
| `molmetal/reports/wf325g_m7_multi_ph_lift/per_cohort.csv` | NEW per-pool co-occurring CSV |
| `metrics/by_metric/diversity_tanimoto.json` | EXISTING — R12 Path A + R12 EV-3 + TODO-180 baselines |
| `metrics/by_metric/diversity_subpocket.json` | EXISTING — DESIGN_ONLY per Wf-318G honest-negative |
| `TODO/completed/82_metric_gap_filling.md` | EXISTING — needs §G4 update (pool-too-small assumption refuted) |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.5 | EXISTING — needs m7 root-cause clarification |
| `TODO/completed/327f16_m7_diversity_investigation.md` | NEW — this document |

---

## §9 Pre-flight gate verification (TODO-207 v2)

Per the Wf-325G closeout pattern (per `TODO/completed/325f_f2a_phase3g_unblock.md` §6), the Wf-327-F16 deliverables are:

1. `molmetal/reports/wf325g_m7_multi_ph_lift/result.json` — Wf-325G measurement result ✅
2. `molmetal/reports/wf325g_m7_multi_ph_lift/per_cohort.csv` — per-cohort CSV ✅
3. `TODO/completed/327f16_m7_diversity_investigation.md` — this investigation document ✅

The deliverables are file-presence checks only (no smoke re-run) so the v2 pre-flight remains CPU-only and < 30 s.

---

## §10 Cross-references

| Reference | Cross-link |
|---|---|
| `metrics/by_metric/diversity_tanimoto.json` | R12 Path A 0.1065 + R12 EV-3 0.1366 + TODO-180 0.0397 baselines |
| `metrics/by_metric/diversity_subpocket.json` | Wf-318G DESIGN_ONLY honest-negative |
| `TODO/completed/82_metric_gap_filling.md` §G4 | **REFUTES** "pool-too-small" assumption — pool ceiling is 0.907 |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.5 | m7 root cause clarification needed |
| `TODO/03_plans/LAMBDA_ROADMAP.md` §1.4 L7 | Multi-PH + PLL end-to-end R² lift (R20-P3) |
| `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` | M24 Multi-PH (R18-P3 GPU retrain — separate workstream) |
| `TODO/completed/325f_f2a_phase3g_unblock.md` | Wf-325 series closeout template (CPU-only, file-presence) |
| `TODO/completed/189_paper_4_FINAL.md` §4.4 | Paper-side m7 anchor: 0.0397 with Tanimoto reward |
| `TODO/completed/wf289_final_draft_signoff.md` | Final-draft headline-metric list (m7 = 0.0397 with +22.5% lift) |
| `mol-metal-cfm-training` | CFM backbone skill (orthogonal to m7 diversity) |
| `mol-metal-multiph-pll` (NEW R20) | Multi-PH + PLL Hilbert regression (orthogonal to m7 diversity) |
| `mol-metal-triangle-closure` (NEW R21) | ASTSR + Hilbert + MCTS 3-way (orthogonal to m7 diversity) |

---

**END OF Wf-327-F16** — Owner Hugo Lee; cross-ref `molmetal/reports/wf325g_m7_multi_ph_lift/result.json` (this run's measurement), `metrics/by_metric/diversity_tanimoto.json` (R12 baselines); last update 2026-09-29.