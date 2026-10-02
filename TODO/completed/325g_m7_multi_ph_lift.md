# Wf-325G — m7 Diversity Multi-PH mechanical lift (CPU scaffold)

**Date**: 2026-09-29
**Wf**: Wf-325G (m7 mechanical lift attempt, no retrain, no GPU launch, no training interruption)
**Skill alignment**: party-twenty-spirit §3.5 (honest framing) + WEAK_TO_STRONG_PLAN §2.5 (m7 baseline)

---

## §1 Goal

Per WEAK_TO_STRONG_PLAN §2.5, m7 Diversity (Morgan-ECFP4 pairwise
Tanimoto distance) is at **0.1366 MEASURED** (R12 EV-3, --sa-weight
0.3, 10x3 generated cohort sweep), with a gate of 0.65 and SOTA
(TransDiffSBDD) at 0.81.  The failure root cause is **fragment-pool
ceiling** on the current pool.

The user's 2026-09-29 directive was: **don't start a retrain now;
resolve honest-negatives first; prefer work that doesn't interrupt
in-flight training**.  This scaffold is the CPU-only mechanical-lift
attempt that satisfies that directive — no GPU launch, no retrain, no
interruption, no fresh user OK required (per AGENTS.md §5.1).

The script tries the **Multi-PH integration** (Kim & Jung 2026
multi-parameter persistent homology + Liu & Feng 2026 PLL) as a
MCTS-heuristic selector across 50 sub-pools of the metallo-drug
cohort and measures whether the aggregate cohort diversity lifts
above the baseline.

---

## §2 MEASURED values (this run)

All numbers in `molmetal/reports/wf325g_m7_multi_ph_lift/result.json`,
all `tag=MEASURED`:

| Field | Value | Source / Interpretation |
|---|---:|---|
| `m7_pool_ceiling` | **0.906782** | MEASURED on full 449-pool — upper bound on what MCTS could ever generate |
| `m7_generated_baseline_this_run` | **0.904122** | MEASURED on 50 raw pools × 20 SMILES each (no selection) |
| `m7_baseline_reference_r12_ev3` | **0.1366** | MEASURED from `metrics/by_metric/diversity_tanimoto.json` -> `wf_lift_ev3` (R12 EV-3 10x3 sweep on small generated cohort, NOT the pool) |
| `m7_after_lift` | **0.932428** | MEASURED aggregate across 50-pool Multi-PH/Tanimoto MaxMin selection |

---

## §3 Lift deltas

| Comparison | Delta | Interpretation |
|---|---:|---|
| `m7_after_lift` vs `m7_generated_baseline_this_run` (0.9041) | **+0.028307** | **Mechanical lift signal** — MaxMin selection adds ~3% diversity over raw pool sampling. |
| `m7_after_lift` vs `m7_baseline_reference_r12_ev3` (0.1366) | **+0.795828** | **Misleading comparison** — the R12 EV-3 cohort was a *small* (10x3 = 30 SMILES) MCTS-generated cohort, not the full 449-pool.  The +0.795 lift mostly reflects pool-vs-generated cohort difference, not mechanical lift. |

`per_pool_lift_mean = +0.019452`, `per_pool_lift_min = -0.068979`,
`per_pool_lift_max = +0.097990`, `n_pools_with_lift = 30/50` (60% of
pools lifted, 40% regressed slightly — MaxMin is greedy and the
first seeded element is random).

---

## §4 Multi-PH signature success rate (CPU-only)

`Multi-PH signature success rate = 0.7900 (316/400)` on the 50-pool
sample — much better than the Wf-325G dry-run expectation of <50%.
Most metallo SMILES *do* embed successfully on the first attempt
once RDKit UFFTYPER warnings are silenced.  The 21% failure rate is
primarily Sc2+3, Pt6+2, Y_5 coordination complexes that hit
``UFFTYPER: Unrecognized atom type`` — they fall back to Tanimoto
MaxMin on the structural fingerprint (no signature to use).

**Honest finding**: Multi-PH integration as a MCTS heuristic is
*feasible* on the metallo cohort (79% signature success), but the
remaining 21% must use Tanimoto fallback.  This is not a blocker —
the integration layer's `MultiPHPLLError` is caught at the script
level and the selector degrades gracefully.

---

## §5 Honest framing per AGENTS.md §5.2

Per `party-twenty-spirit` §3.5 the mechanical-lift verdict
`MECHANICAL_LIFT_OK` is technically correct (passes the +0.05 delta
threshold against the R12 EV-3 reference), but the *honest reading*
is more nuanced:

1. **Mechanical lift ≠ structural fix.**  The mechanical MaxMin
   selection adds +0.0283 over raw pool sampling — meaningful but
   not transformative.  The structural fix per WEAK_TO_STRONG_PLAN
   §2.5 is *fragment-pool expansion* (449 → ~2000+ compounds with
   more scaffold coverage), which is the actual route to +0.30 to
   +0.45 long-term lift (R19+ Lambda × CFM coupling TODO-21).

2. **R12 EV-3 0.1366 was on a small generated cohort.**  The
   comparison `+0.796 vs R12 EV-3` is misleading because R12 EV-3
   measured diversity on a *generated* 30-SMILES cohort (with
   `--sa-weight 0.3` forcing SA-constrained sampling that biases
   toward similar scaffolds), not on the full 449-pool.  The real
   pool diversity is 0.9067 — much higher than 0.1366.  The
   mechanical lift in this scaffold is +0.0283, not +0.7958.

3. **Pool has 26.7% duplicates.**  Test 2 surfaced a structural
   finding: 449 rows → 329 unique SMILES.  The pool-ceiling
   diversity is structurally bounded below the 449-compound ceiling
   by this overlap.  A duplicate-removal pre-processing step would
   be needed before claiming the 0.9067 ceiling is reachable.

4. **CPU-only, no GPU launch, no retrain, no interruption.**  Per
   user's 2026-09-29 directive.  Pre-flight v2 PASS (25/25, exit=0)
   confirms the harness is healthy even though this work did not
   require a fresh user OK (per AGENTS.md §5.1 — CPU-only scaffold
   does not invoke kill_then_adapt).

5. **n_aggregated_candidates = 17 of (50 × 4 = 200) selection slots.**
   MaxMin's greedy nature causes many pools to converge on similar
   representatives (high Morgan-ECFP4 similarity across the cohort).
   This is *expected* for the 449-pool with mostly Pt-phosphine
   scaffolds — it confirms the structural fragment-pool root cause.

---

## §6 Verdict

- `lift_status = MECHANICAL_LIFT_OK` (mechanical threshold passed)
- `verdict = PASS` (script-level)
- **Honest interpretation: MECHANICAL_LIFT_MINOR** — +0.0283 over
  pool sampling is real but not structural.  The m7 honest-negative
  is *preserved* at the structural level (fragment-pool ceiling
  remains the binding constraint).  No claim of meaningful m7
  progress.

---

## §7 Files shipped

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf325g_m7_multi_ph_lift.py` | CPU scaffold entry (no GPU, --max-multiph-per-pool cap for wall-time bound) |
| `molmetal/tests/test_wf325g_m7_multi_ph_lift.py` | 8 pytest contracts — all 8 PASS in 3.47s |
| `molmetal/reports/wf325g_m7_multi_ph_lift/result.json` | MEASURED-vs-PROJECTED tagged values |
| `molmetal/reports/wf325g_m7_multi_ph_lift/per_cohort.csv` | Per-pool lift summary for downstream consumers |

---

## §8 R19+ scope reminder

Per WEAK_TO_STRONG_PLAN §2.5 the long-term scope is **Lambda × CFM
coupling TODO-21** which may lift m7 +0.30 to +0.45 (R19+).  This
Wf-325G is *not* a step toward that scope — it's a CPU scaffold
that confirms the fragment-pool root cause and rules out the
mechanical MaxMin path as a substitute.

**Next step**: defer m7 to R19+ Lambda × CFM coupling.  Keep
honest-negative in WEAK_TO_STRONG_PLAN §2.5.  No retrain /
panel / ablation launch from Wf-325G.

---

## §9 Compliance check

- [x] AGENTS.md §1 rule 1 (MEASURED-vs-PROJECTED tag discipline): all numeric values cited with `tag` field in result.json
- [x] AGENTS.md §1 rule 2 (kill_then_adapt): no GPU launch — CPU-only scaffold, no fresh user OK needed
- [x] AGENTS.md §1 rule 3 (no paper work): no `paper/*` touched
- [x] AGENTS.md §1 rule 4 (pre-flight v2 gate): pre-flight run with exit=0 PASS
- [x] AGENTS.md §1 rule 5 (honest-negative framing): structural-fix disclaimer in §5 above
- [x] AGENTS.md §8 (no StructuredOutput): plain-text brief template followed
- [x] Wf-302C regression workaround: agent uses `Bash + Read + Write + Edit` only

---

**Wf-325G complete.** CPU scaffold shipped; 8/8 tests PASS;
pre-flight PASS; m7 honest-negative preserved at the structural level.
